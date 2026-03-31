"""Flower ClientApp: Local LoRA training on GLUE tasks using HF Trainer."""

import numpy as np
import torch
from evaluate import load as load_metric
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from bert.dataset import load_data, get_num_labels
from bert.models import get_model, get_parameters, set_parameters


def compute_metrics(eval_pred):
    """Compute accuracy for GLUE classification tasks."""
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class FlowerClient(NumPyClient):
    def __init__(self, net, train_dataset, eval_dataset, tokenizer, data_collator,
                 local_epochs, learning_rate, batch_size, grad_accum_steps):
        self.net = net
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        # Use round-level cosine annealing LR from server config
        lr = config.get("learning_rate", self.learning_rate)

        training_args = TrainingArguments(
            output_dir="./tmp_fl_client",
            num_train_epochs=self.local_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            learning_rate=lr,
            weight_decay=0.01,
            lr_scheduler_type="constant",  # Server handles LR scheduling
            logging_steps=10,
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

        return (
            get_parameters(self.net),
            len(self.train_dataset),
            {"train_loss": results.training_loss},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.eval_dataset, self.data_collator,
                              self.batch_size, self.device)
        return float(loss), len(self.eval_dataset), {"accuracy": accuracy}


def test(net, eval_dataset, data_collator, batch_size, device):
    """Evaluate the model and return loss + accuracy."""
    metric = load_metric("accuracy")
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
    accuracy = metric.compute()["accuracy"]
    return avg_loss, accuracy


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

    num_labels = get_num_labels(task_name)

    # Load data (returns HF Datasets, not DataLoaders)
    train_dataset, eval_dataset, tokenizer, data_collator = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        task_name=task_name,
        model_name=model_name,
        dirichlet_alpha=dirichlet_alpha,
        max_seq_length=max_seq_length,
    )

    # Load model
    net = get_model(
        model_name=model_name,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=target_modules,
    )

    return FlowerClient(
        net, train_dataset, eval_dataset, tokenizer, data_collator,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
    ).to_client()


app = ClientApp(client_fn=client_fn)
