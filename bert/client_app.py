"""Flower ClientApp: Local LoRA training on GLUE tasks."""

import torch
from evaluate import load as load_metric
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from bert.dataset import load_data, get_num_labels
from bert.models import get_model, get_parameters, set_parameters


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs, learning_rate,
                 grad_accum_steps):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.grad_accum_steps = grad_accum_steps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            lr=self.learning_rate,
            grad_accum_steps=self.grad_accum_steps,
            device=self.device,
        )
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}


def train(net, trainloader, epochs, lr, grad_accum_steps, device):
    """Train the model with AdamW + cosine scheduler."""
    optimizer = AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(trainloader))

    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(trainloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(trainloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


def test(net, testloader, device):
    """Evaluate the model and return loss + accuracy."""
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

    # Load data
    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        task_name=task_name,
        model_name=model_name,
        dirichlet_alpha=dirichlet_alpha,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
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
        net, trainloader, testloader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        grad_accum_steps=grad_accum_steps,
    ).to_client()


app = ClientApp(client_fn=client_fn)
