"""Centralized training script for RoBERTa + LoRA on GLUE tasks.

Shares the same model/dataset utilities as the federated learning code.
Usage:
    cd /data/experiment/exp-fed/BERT/bert
    python centralized_learning/train.py --task sst2
    python centralized_learning/train.py --task qnli --lora-r 16
    python centralized_learning/train.py --task mnli --lora-r 16
"""

import argparse
import os
import sys

import torch
from evaluate import load as load_metric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent dir to path so we can import bert package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bert.dataset import get_num_labels, load_centralized_data
from bert.models import get_model


def train_one_epoch(net, trainloader, optimizer, scheduler, grad_accum_steps, device):
    """Train for one epoch with gradient accumulation."""
    net.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(trainloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = net(**batch)
        loss = outputs.loss / grad_accum_steps
        loss.backward()
        total_loss += outputs.loss.item()
        num_batches += 1

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(trainloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(net, testloader, device):
    """Evaluate and return loss + accuracy."""
    metric = load_metric("accuracy")
    total_loss = 0.0
    num_batches = 0
    net.eval()

    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = metric.compute()["accuracy"]
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Centralized LoRA training on GLUE")
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["sst2", "qnli", "mnli", "qqp", "rte"])
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-modules", type=str, default="query,value")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="centralized_learning/checkpoints")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_modules = args.lora_target_modules.split(",")
    num_labels = get_num_labels(args.task)

    # Load model
    print(f"Loading {args.model_name} with LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    net = get_model(args.model_name, num_labels, args.lora_r, args.lora_alpha, target_modules)
    net.to(device)

    # Load data
    print(f"Loading {args.task} dataset...")
    trainloader, testloader = load_centralized_data(
        task_name=args.task,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}")

    # Optimizer + scheduler
    optimizer = AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    total_steps = args.epochs * len(trainloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    save_dir = os.path.join(args.output_dir, f"{args.task}_r{args.lora_r}")
    os.makedirs(save_dir, exist_ok=True)

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(net, trainloader, optimizer, scheduler,
                                     args.grad_accum_steps, device)
        eval_loss, accuracy = evaluate(net, testloader, device)

        print(f"Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}, accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_dir = os.path.join(save_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            net.save_pretrained(best_dir)
            print(f"  -> New best accuracy: {accuracy:.4f}, saved to {best_dir}")

        if epoch % args.save_every == 0:
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            net.save_pretrained(epoch_dir)

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
