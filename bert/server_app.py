"""Flower ServerApp: Initialize model + strategy selection."""

import torch
from evaluate import load as load_metric
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader

from bert.dataset import get_num_labels, load_centralized_data
from bert.models import cosine_annealing, get_model, get_parameters, get_parameter_keys, set_parameters, set_seed
from bert.strategy import FedAvgStrategy
from bert.fedsa_strategy import FedSALoRAStrategy


def _eval_on_loader(net, testloader, device, task_name):
    """Evaluate model on a DataLoader, return (loss, metrics_dict)."""
    metric = load_metric("glue", task_name)
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
    results = metric.compute()  # e.g. {"accuracy": ..., "f1": ...} for QQP
    return avg_loss, results


def get_evaluate_fn(model_name, task_name, num_labels, lora_r, lora_alpha,
                    target_modules, max_seq_length, batch_size, lora_dropout=0.1):
    """Return a server-side evaluation function."""
    # Pre-load eval data once
    _, eval_dataset, _, data_collator = load_centralized_data(
        task_name=task_name,
        model_name=model_name,
        max_seq_length=max_seq_length,
    )
    testloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)

    # MNLI: also prepare mismatched validation set
    testloader_mm = None
    if task_name == "mnli":
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        ds_mm = load_dataset("nyu-mll/glue", "mnli", split="validation_mismatched")
        ds_mm = ds_mm.map(lambda x: tok(x["premise"], x["hypothesis"],
                                         truncation=True, max_length=max_seq_length), batched=True)
        cols_remove = [c for c in ds_mm.column_names
                       if c not in ("input_ids", "attention_mask", "token_type_ids", "label", "labels")]
        ds_mm = ds_mm.remove_columns(cols_remove)
        if "label" in ds_mm.column_names:
            ds_mm = ds_mm.rename_column("label", "labels")
        ds_mm.set_format("torch")
        testloader_mm = DataLoader(ds_mm, batch_size=batch_size, collate_fn=data_collator)

    def evaluate(server_round, parameters, config):
        if server_round == 0:
            return 0.0, {}

        net = get_model(model_name, num_labels, lora_r, lora_alpha, target_modules, lora_dropout)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        set_parameters(net, parameters)

        avg_loss, eval_results = _eval_on_loader(net, testloader, device, task_name)
        metrics = dict(eval_results)  # e.g. {"accuracy": ..., "f1": ...} for QQP

        log_msg = f"[Server] Round {server_round}: loss={avg_loss:.4f}"
        for k, v in eval_results.items():
            log_msg += f", {k}={v:.4f}"

        # MNLI: evaluate mismatched
        if testloader_mm is not None:
            _, eval_results_mm = _eval_on_loader(net, testloader_mm, device, task_name)
            for k, v in eval_results_mm.items():
                metrics[f"{k}_mm"] = v
                log_msg += f", {k}_mm={v:.4f}"

        # Clean up
        del net
        torch.cuda.empty_cache()

        print(log_msg)
        return avg_loss, metrics

    return evaluate


def server_fn(context: Context):
    """Configure the server with strategy selection."""
    cfg = context.run_config

    # Read config
    num_rounds = int(cfg["num-server-rounds"])
    fraction_fit = float(cfg["fraction-fit"])
    model_name = str(cfg["model-name"])
    task_name = str(cfg["task-name"])
    lora_r = int(cfg["lora-r"])
    lora_alpha = int(cfg["lora-alpha"])
    target_modules = str(cfg["lora-target-modules"]).split(",")
    aggregation_mode = str(cfg["aggregation-mode"])
    max_seq_length = int(cfg["max-seq-length"])
    batch_size = int(cfg["batch-size"])
    lora_dropout = float(cfg["lora-dropout"])
    seed = int(cfg["seed"])

    set_seed(seed)

    num_labels = get_num_labels(task_name)

    # Initialize global model to get initial parameters and key ordering
    net = get_model(model_name, num_labels, lora_r, lora_alpha, target_modules, lora_dropout)
    initial_weights = get_parameters(net)
    lora_param_keys = get_parameter_keys(net)
    initial_parameters = ndarrays_to_parameters(initial_weights)
    del net

    learning_rate = float(cfg["learning-rate"])

    # Build on_fit_config_fn with round-level cosine annealing
    def fit_config(server_round: int):
        lr = cosine_annealing(
            current_round=server_round,
            total_round=num_rounds,
            lrate_max=learning_rate,
            lrate_min=1e-6,
        )
        return {"current_round": server_round, "learning_rate": lr}

    # Build evaluate function
    evaluate_fn = get_evaluate_fn(
        model_name, task_name, num_labels, lora_r, lora_alpha,
        target_modules, max_seq_length, batch_size, lora_dropout,
    )

    # Create strategy based on aggregation mode
    if aggregation_mode == "full":
        strategy = FedAvgStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
        )
    else:
        strategy = FedSALoRAStrategy(
            aggregation_mode=aggregation_mode,
            lora_param_keys=lora_param_keys,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
