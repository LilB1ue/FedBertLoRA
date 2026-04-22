"""Flower ClientApp: Local LoRA training on GLUE tasks using HF Trainer."""

import os

import numpy as np
import torch
from evaluate import load as load_metric
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from bert.dataset import load_data, get_num_labels
from bert.models import get_model, get_parameters, set_parameters, set_seed, freeze_lora_a




class FlowerClient(NumPyClient):
    def __init__(self, net, train_dataset, eval_dataset, tokenizer, data_collator,
                 local_epochs, learning_rate, batch_size, grad_accum_steps,
                 partition_id: int = 0, weight_decay: float = 0.01,
                 lr_scheduler_type: str = "constant", logging_steps: int = 10,
                 task_name: str = "sst2", checkpoint_dir: str = None,
                 seed: int = 42, log_dir: str = "logs",
                 aggregation_mode: str = "fedavg",
                 dirichlet_alpha: float = 0.5):
        self.net = net
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.partition_id = partition_id
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.logging_steps = logging_steps
        self.task_name = task_name
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.log_dir = log_dir
        self.aggregation_mode = aggregation_mode
        self.dirichlet_alpha = dirichlet_alpha
        # Matches server_app.py batch_dir layout:
        #   logs/<ts>_<mode>_a<alpha>/<task>_<mode>_a<alpha>/
        self._alpha_tag = f"_a{float(dirichlet_alpha)}"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        current_round = config.get("current_round", 0)
        log_timestamp = config.get("log_timestamp", "")

        # Save received (post-aggregation) adapter for non-fedavg strategies
        if self.aggregation_mode != "fedavg" and log_timestamp:
            batch_dir = f"{log_timestamp}_{self.aggregation_mode}{self._alpha_tag}"
            run_subdir = f"{self.task_name}_{self.aggregation_mode}{self._alpha_tag}"
            recv_dir = os.path.join(
                self.log_dir, batch_dir, run_subdir,
                "received_checkpoints",
                f"round_{current_round}", f"client_{self.partition_id}",
            )
            os.makedirs(recv_dir, exist_ok=True)
            self.net.save_pretrained(recv_dir)

        # Use round-level cosine annealing LR from server config
        lr = config.get("learning_rate", self.learning_rate)

        training_args = TrainingArguments(
            output_dir="./tmp_fl_client",
            num_train_epochs=self.local_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            learning_rate=lr,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            seed=self.seed,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.net,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
        )

        results = trainer.train()

        # Local eval on the same model (before aggregation) for overfitting tracking
        eval_loss, eval_metrics = test(self.net, self.eval_dataset, self.data_collator,
                                       self.batch_size, self.device, self.task_name)

        fit_metrics = {
            "partition_id": self.partition_id,
            "train_loss": results.training_loss,
            "eval_loss": eval_loss,
            "train_samples": len(self.train_dataset),
            "eval_samples": len(self.eval_dataset),
        }
        for k, v in eval_metrics.items():
            fit_metrics[f"eval_{k}"] = v

        # Save per-client LoRA checkpoint (trained, before aggregation) — all strategies
        if log_timestamp:
            batch_dir = f"{log_timestamp}_{self.aggregation_mode}{self._alpha_tag}"
            run_subdir = f"{self.task_name}_{self.aggregation_mode}{self._alpha_tag}"
            ckpt_dir = os.path.join(
                self.log_dir, batch_dir, run_subdir, "client_checkpoints"
            )
        else:
            ckpt_dir = self.checkpoint_dir
        if ckpt_dir:
            save_path = os.path.join(ckpt_dir, f"round_{current_round}", f"client_{self.partition_id}")
            os.makedirs(save_path, exist_ok=True)
            self.net.save_pretrained(save_path)

        params = get_parameters(self.net)
        del self.net
        torch.cuda.empty_cache()

        return (
            params,
            len(self.train_dataset),
            fit_metrics,
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, eval_metrics = test(self.net, self.eval_dataset, self.data_collator,
                                  self.batch_size, self.device, self.task_name)
        del self.net
        torch.cuda.empty_cache()
        metrics = {**eval_metrics, "partition_id": self.partition_id}
        return float(loss), len(self.eval_dataset), metrics


def test(net, eval_dataset, data_collator, batch_size, device, task_name="sst2"):
    """Evaluate the model and return loss + task-specific metrics."""
    metric = load_metric("glue", task_name)
    testloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)
    total_loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    results = metric.compute()  # {"accuracy": ...} or {"accuracy": ..., "f1": ...} for QQP
    return avg_loss, results


def client_fn(context: Context):
    """Construct a FlowerClient from Flower context."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read config
    cfg = context.run_config
    model_name = cfg["model-name"]
    task_name = cfg["task-name"]
    lora_r = int(cfg["lora-r"])
    lora_alpha = int(cfg["lora-alpha"])
    target_modules = str(cfg["lora-target-modules"]).split(",")
    dirichlet_alpha = float(cfg["dirichlet-alpha"])
    max_seq_length = int(cfg["max-seq-length"])
    batch_size = int(cfg["batch-size"])
    local_epochs = int(cfg["local-epochs"])
    learning_rate = float(cfg["learning-rate"])
    grad_accum_steps = int(cfg["grad-accum-steps"])
    weight_decay = float(cfg["weight-decay"])
    lora_dropout = float(cfg["lora-dropout"])
    lr_scheduler_type = str(cfg["lr-scheduler-type"])
    test_split_ratio = float(cfg["test-split-ratio"])
    seed = int(cfg["seed"])
    logging_steps = int(cfg["logging-steps"])
    log_dir = str(cfg.get("log-dir", "logs"))
    aggregation_mode = str(cfg.get("aggregation-mode", "fedavg"))
    log_timestamp = str(cfg.get("log-timestamp", ""))
    alpha_tag = f"_a{dirichlet_alpha}"
    batch_dir = f"{log_timestamp}_{aggregation_mode}{alpha_tag}"
    run_subdir = f"{task_name}_{aggregation_mode}{alpha_tag}"
    checkpoint_dir = os.path.join(log_dir, batch_dir, run_subdir, "client_checkpoints")

    set_seed(seed)

    num_labels = get_num_labels(task_name)

    # Load data (returns HF Datasets, not DataLoaders)
    train_dataset, eval_dataset, tokenizer, data_collator = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        task_name=task_name,
        model_name=model_name,
        dirichlet_alpha=dirichlet_alpha,
        max_seq_length=max_seq_length,
        test_size=test_split_ratio,
        seed=seed,
    )

    # Load model
    net = get_model(
        model_name=model_name,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=target_modules,
        lora_dropout=lora_dropout,
    )

    # FFA-LoRA: freeze A matrices on client side (only train B)
    if aggregation_mode == "ffa":
        freeze_lora_a(net)

    return FlowerClient(
        net, train_dataset, eval_dataset, tokenizer, data_collator,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        partition_id=partition_id,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        task_name=task_name,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        log_dir=log_dir,
        aggregation_mode=aggregation_mode,
        dirichlet_alpha=dirichlet_alpha,
    ).to_client()


app = ClientApp(client_fn=client_fn)
