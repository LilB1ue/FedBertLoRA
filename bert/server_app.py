"""Flower ServerApp: Initialize model + FedSA-LoRA strategy."""

import torch
from evaluate import load as load_metric
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from bert.dataset import get_num_labels, load_centralized_data
from bert.models import get_model, get_parameters, get_parameter_keys, set_parameters
from bert.strategy import FedSALoRAStrategy


def get_evaluate_fn(model_name, task_name, num_labels, lora_r, lora_alpha,
                    target_modules, max_seq_length, batch_size):
    """Return a server-side evaluation function."""

    def evaluate(server_round, parameters, config):
        if server_round == 0:
            return 0.0, {}

        net = get_model(model_name, num_labels, lora_r, lora_alpha, target_modules)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        set_parameters(net, parameters)

        _, testloader = load_centralized_data(
            task_name=task_name,
            model_name=model_name,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )

        metric = load_metric("accuracy")
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

        # Clean up
        del net
        torch.cuda.empty_cache()

        print(f"[Server] Round {server_round}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
        return avg_loss, {"accuracy": accuracy}

    return evaluate


def server_fn(context: Context):
    """Configure the server with FedSA-LoRA strategy."""
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

    num_labels = get_num_labels(task_name)

    # Initialize global model to get initial parameters and key ordering
    net = get_model(model_name, num_labels, lora_r, lora_alpha, target_modules)
    initial_weights = get_parameters(net)
    lora_param_keys = get_parameter_keys(net)
    initial_parameters = ndarrays_to_parameters(initial_weights)
    del net

    # Build on_fit_config_fn
    def fit_config(server_round: int):
        return {"current_round": server_round}

    # Build evaluate function
    evaluate_fn = get_evaluate_fn(
        model_name, task_name, num_labels, lora_r, lora_alpha,
        target_modules, max_seq_length, batch_size,
    )

    # Create strategy
    strategy = FedSALoRAStrategy(
        aggregation_mode=aggregation_mode,
        lora_param_keys=lora_param_keys,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # Server-side eval only
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
