#!/bin/bash
# Run centralized LoRA training on all GLUE tasks.
# Usage:
#   cd /data/experiment/exp-fed/BERT/bert
#   bash centralized_learning/run_all.sh
#   bash centralized_learning/run_all.sh --wandb   # with wandb logging

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Pass extra args (e.g. --wandb) from command line
EXTRA_ARGS="$@"

# Task configs: task_name lora_r
# r=8 for all tasks (LoRA paper shows r=8 is sufficient for RoBERTa-large on GLUE)
TASKS=(
    "sst2 8"
    "qnli 8"
    "mnli 8"
    "qqp 8"
    "rte 8"
)

echo "=========================================="
echo "Centralized LoRA Training - All GLUE Tasks"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

for task_config in "${TASKS[@]}"; do
    read -r TASK LORA_R <<< "$task_config"

    echo ""
    echo ">>> Starting ${TASK} (LoRA r=${LORA_R})"
    echo ">>> $(date)"

    python centralized_learning/train.py \
        --task "$TASK" \
        --lora-r "$LORA_R" \
        --epochs 10 \
        --early-stopping-patience 3 \
        $EXTRA_ARGS

    echo ">>> Finished ${TASK} at $(date)"
    echo ""
done

echo "=========================================="
echo "All tasks complete!"
echo "Logs: centralized_learning/logs/"
echo "=========================================="
