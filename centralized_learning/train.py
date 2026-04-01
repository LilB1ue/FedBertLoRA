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
from datetime import datetime

import numpy as np
from evaluate import load as load_metric
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

# Add parent dir to path so we can import bert package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bert.dataset import get_num_labels, load_centralized_data
from bert.models import get_model


class FileLoggingCallback(TrainerCallback):
    """Write epoch-level metrics to a log file."""

    def __init__(self, log_path):
        self.log_path = log_path
        with open(log_path, "w") as f:
            f.write("epoch\ttrain_loss\teval_loss\teval_accuracy\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = int(state.epoch)
        train_loss = state.log_history[-2].get("loss", "") if len(state.log_history) >= 2 else ""
        eval_loss = metrics.get("eval_loss", "")
        eval_acc = metrics.get("eval_accuracy", "")
        with open(self.log_path, "a") as f:
            f.write(f"{epoch}\t{train_loss}\t{eval_loss}\t{eval_acc}\n")


def compute_metrics(eval_pred):
    """Compute accuracy for GLUE classification tasks."""
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    parser = argparse.ArgumentParser(description="Centralized LoRA training on GLUE")
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["sst2", "qnli", "mnli", "qqp", "rte"])
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-modules", type=str, default="query,key,value,dense")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="centralized_learning/checkpoints")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop after N epochs without improvement")
    parser.add_argument("--log-dir", type=str, default="centralized_learning/logs",
                        help="Directory to save per-epoch training logs")
    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="bert-centralized",
                        help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Wandb run name (default: auto-generated from task/lora config)")
    args = parser.parse_args()

    target_modules = args.lora_target_modules.split(",")
    num_labels = get_num_labels(args.task)

    # Wandb setup
    if args.wandb:
        import wandb
        os.environ["WANDB_LOG_MODEL"] = "false"
        ts = datetime.now().strftime("%m%d_%H%M")
        run_name = args.wandb_run_name or f"{args.task}_r{args.lora_r}_{ts}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        )

    # Load model
    print(f"Loading {args.model_name} with LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    net = get_model(args.model_name, num_labels, args.lora_r, args.lora_alpha, target_modules)

    # Load data
    print(f"Loading {args.task} dataset...")
    train_dataset, eval_dataset, tokenizer, data_collator = load_centralized_data(
        task_name=args.task,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    save_dir = os.path.join(args.output_dir, f"{args.task}_r{args.lora_r}")

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb" if args.wandb else "none",
        remove_unused_columns=False,
    )

    # Setup per-epoch file logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.task}_r{args.lora_r}.tsv")
    file_logger = FileLoggingCallback(log_path)

    trainer = Trainer(
        model=net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            file_logger,
        ],
    )

    trainer.train()

    # Save best model
    best_dir = os.path.join(save_dir, "best")
    trainer.save_model(best_dir)

    # Final evaluation
    metrics = trainer.evaluate()
    print(f"\nTraining complete. Final eval (matched): {metrics}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # MNLI: also evaluate on mismatched validation set
    if args.task == "mnli":
        from bert.dataset import load_centralized_data as _load
        _, eval_mm, _, _ = _load(task_name="mnli", model_name=args.model_name,
                                  max_seq_length=args.max_seq_length)
        # load_centralized_data returns validation_matched by default, load mismatched manually
        from datasets import load_dataset
        ds_mm = load_dataset("nyu-mll/glue", "mnli", split="validation_mismatched")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name)
        ds_mm = ds_mm.map(lambda x: tok(x["premise"], x["hypothesis"],
                                         truncation=True, max_length=args.max_seq_length), batched=True)
        cols_remove = [c for c in ds_mm.column_names
                       if c not in ("input_ids", "attention_mask", "token_type_ids", "label", "labels")]
        ds_mm = ds_mm.remove_columns(cols_remove)
        if "label" in ds_mm.column_names:
            ds_mm = ds_mm.rename_column("label", "labels")
        ds_mm.set_format("torch")

        metrics_mm = trainer.evaluate(eval_dataset=ds_mm, metric_key_prefix="eval_mm")
        print(f"  MNLI mismatched: {metrics_mm}")
        trainer.log_metrics("eval_mm", metrics_mm)
        trainer.save_metrics("eval_mm", metrics_mm)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
