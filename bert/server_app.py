"""Flower ServerApp: Initialize model + strategy selection."""

import os
from datetime import datetime

import torch
from evaluate import load as load_metric
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader

from bert.dataset import get_num_labels, load_centralized_data
from bert.models import cosine_annealing, get_model, get_parameters, get_parameter_keys, set_parameters, set_seed
from bert.strategy import FedAvgStrategy
from bert.fedsa_strategy import FedSALoRAStrategy


def get_metrics_aggregation_fn(log_path, phase, use_wandb=False):
    """Return a metrics aggregation function that logs per-client metrics to TSV.

    Args:
        log_path: TSV file path to write per-client metrics.
        phase: "fit" or "evaluate" — used for wandb prefix.
        use_wandb: Whether to also log aggregated metrics to wandb.
    """
    _round_counter = [0]
    _header_written = [False]

    def aggregate(metrics_list):
        """metrics_list: List[(num_examples, metrics_dict)] from each client."""
        _round_counter[0] += 1
        current_round = _round_counter[0]

        # Determine columns from first client's metrics (exclude partition_id)
        sample_metrics = metrics_list[0][1]
        metric_keys = [k for k in sorted(sample_metrics.keys()) if k != "partition_id"]

        # Write header on first call
        if not _header_written[0]:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                f.write("round\tpartition_id\tnum_examples\t" + "\t".join(metric_keys) + "\n")
            _header_written[0] = True

        # Write per-client rows (sorted by partition_id)
        sorted_metrics = sorted(metrics_list, key=lambda x: x[1].get("partition_id", 0))
        with open(log_path, "a") as f:
            for num_examples, client_metrics in sorted_metrics:
                pid = client_metrics.get("partition_id", "?")
                values = [str(client_metrics.get(k, "")) for k in metric_keys]
                f.write(f"{current_round}\t{pid}\t{num_examples}\t" + "\t".join(values) + "\n")

        # Compute weighted average + stats for return
        total = sum(n for n, _ in metrics_list)
        aggregated = {}
        for key in metric_keys:
            vals = [(n, m.get(key)) for n, m in metrics_list if isinstance(m.get(key), (int, float))]
            if vals and total > 0:
                raw_vals = [v for _, v in vals]
                aggregated[key] = sum(n * v for n, v in vals) / total
                aggregated[f"{key}_std"] = float(torch.tensor(raw_vals, dtype=torch.float64).std())
                aggregated[f"{key}_min"] = min(raw_vals)
                aggregated[f"{key}_max"] = max(raw_vals)

        # Log to wandb
        if use_wandb:
            import wandb
            wandb_metrics = {f"{phase}/{k}": v for k, v in aggregated.items()}
            wandb_metrics["round"] = current_round
            wandb.log(wandb_metrics, step=current_round)

        return aggregated

    return aggregate


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
                    target_modules, max_seq_length, batch_size, lora_dropout=0.1,
                    server_log_path=None, use_wandb=False, checkpoint_dir=None):
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

    _server_header_written = [False]

    def evaluate(server_round, parameters, config):
        if server_round == 0:
            return 0.0, {}

        net = get_model(model_name, num_labels, lora_r, lora_alpha, target_modules, lora_dropout)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        set_parameters(net, parameters)

        avg_loss, eval_results = _eval_on_loader(net, testloader, device, task_name)
        metrics = dict(eval_results)
        metrics["loss"] = avg_loss

        log_msg = f"[Server] Round {server_round}: loss={avg_loss:.4f}"
        for k, v in eval_results.items():
            log_msg += f", {k}={v:.4f}"

        # MNLI: evaluate mismatched
        if testloader_mm is not None:
            _, eval_results_mm = _eval_on_loader(net, testloader_mm, device, task_name)
            for k, v in eval_results_mm.items():
                metrics[f"{k}_mm"] = v
                log_msg += f", {k}_mm={v:.4f}"

        # Save global LoRA checkpoint
        if checkpoint_dir:
            save_path = os.path.join(checkpoint_dir, f"round_{server_round}")
            os.makedirs(save_path, exist_ok=True)
            net.save_pretrained(save_path)

        # Clean up
        del net
        torch.cuda.empty_cache()

        print(log_msg)

        # Log to wandb
        if use_wandb:
            import wandb
            wandb_metrics = {f"server/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=server_round)

        # Write server eval log
        if server_log_path:
            metric_keys = [k for k in sorted(metrics.keys())]
            if not _server_header_written[0]:
                os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
                with open(server_log_path, "w") as f:
                    f.write("round\t" + "\t".join(metric_keys) + "\n")
                _server_header_written[0] = True
            with open(server_log_path, "a") as f:
                values = [str(metrics.get(k, "")) for k in metric_keys]
                f.write(f"{server_round}\t" + "\t".join(values) + "\n")

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
    log_dir = str(cfg.get("log-dir", "logs"))
    wandb_enabled = bool(cfg.get("wandb-enabled", False))
    wandb_project = str(cfg.get("wandb-project", "bert-federated"))
    num_clients = int(cfg.get("num-clients", 0))
    learning_rate = float(cfg["learning-rate"])

    set_seed(seed)

    # Init wandb (single run on server side)
    if wandb_enabled:
        import wandb
        ts = datetime.now().strftime("%m%d_%H%M")
        run_name = f"{task_name}_{aggregation_mode}_c{num_clients}_{ts}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "task_name": task_name,
                "model_name": model_name,
                "aggregation_mode": aggregation_mode,
                "num_server_rounds": num_rounds,
                "num_clients": num_clients,
                "fraction_fit": fraction_fit,
                "dirichlet_alpha": float(cfg.get("dirichlet-alpha", 0.5)),
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "grad_accum_steps": int(cfg.get("grad-accum-steps", 1)),
                "local_epochs": int(cfg.get("local-epochs", 1)),
                "max_seq_length": max_seq_length,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_target_modules": target_modules,
                "lora_dropout": lora_dropout,
                "weight_decay": float(cfg.get("weight-decay", 0.01)),
                "lr_scheduler_type": str(cfg.get("lr-scheduler-type", "constant")),
                "seed": seed,
            },
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        )

    num_labels = get_num_labels(task_name)

    # Initialize global model to get initial parameters and key ordering
    net = get_model(model_name, num_labels, lora_r, lora_alpha, target_modules, lora_dropout)
    initial_weights = get_parameters(net)
    lora_param_keys = get_parameter_keys(net)
    initial_parameters = ndarrays_to_parameters(initial_weights)
    del net

    # Setup log paths (logs/<timestamp>/<task>_<strategy>/)
    log_timestamp = str(cfg.get("log-timestamp", "")) or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = os.path.join(log_dir, log_timestamp, f"{task_name}_{aggregation_mode}")
    fit_log_path = os.path.join(log_subdir, "fit_metrics.tsv")
    eval_log_path = os.path.join(log_subdir, "eval_metrics.tsv")
    server_log_path = os.path.join(log_subdir, "server_eval.tsv")
    global_ckpt_dir = os.path.join(log_subdir, "global_checkpoints")

    # Build on_fit_config_fn with round-level cosine annealing
    def fit_config(server_round: int):
        lr = cosine_annealing(
            current_round=server_round,
            total_round=num_rounds,
            lrate_max=learning_rate,
            lrate_min=1e-6,
        )
        return {"current_round": server_round, "learning_rate": lr, "log_timestamp": log_timestamp}

    # Build evaluate function
    evaluate_fn = get_evaluate_fn(
        model_name, task_name, num_labels, lora_r, lora_alpha,
        target_modules, max_seq_length, batch_size, lora_dropout,
        server_log_path=server_log_path,
        use_wandb=wandb_enabled,
        checkpoint_dir=global_ckpt_dir,
    )

    # Metrics aggregation functions (log per-client metrics to TSV + optional wandb)
    fit_metrics_agg = get_metrics_aggregation_fn(fit_log_path, "fit", use_wandb=wandb_enabled)
    eval_metrics_agg = get_metrics_aggregation_fn(eval_log_path, "evaluate", use_wandb=wandb_enabled)

    # Create strategy based on aggregation mode
    if aggregation_mode == "fedavg":
        strategy = FedAvgStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=fit_metrics_agg,
            evaluate_metrics_aggregation_fn=eval_metrics_agg,
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
            fit_metrics_aggregation_fn=fit_metrics_agg,
            evaluate_metrics_aggregation_fn=eval_metrics_agg,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
