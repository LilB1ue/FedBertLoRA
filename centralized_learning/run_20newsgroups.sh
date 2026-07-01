#!/bin/bash
# Run centralized LoRA training on 20 Newsgroups.
#
# Usage: bash centralized_learning/run_20newsgroups.sh [epochs] [batch_size] [max_seq_length] [seed] [learning_rate] [wandb]
#   e.g. bash centralized_learning/run_20newsgroups.sh
#        bash centralized_learning/run_20newsgroups.sh 10 16 256 42 1e-4 true

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EPOCHS="${1:-10}"
BATCH_SIZE="${2:-16}"
MAX_SEQ_LENGTH="${3:-256}"
SEED="${4:-42}"
LEARNING_RATE="${5:-1e-4}"
WANDB_ENABLED="${6:-false}"

TASK="20newsgroups"
MODEL_NAME="roberta-large"
LORA_R="8"
LORA_ALPHA="16"
LORA_TARGET_MODULES="query,key,value,dense"
LORA_DROPOUT="0.1"
GRAD_ACCUM_STEPS="4"
WEIGHT_DECAY="0.01"
LR_SCHEDULER_TYPE="cosine"
EARLY_STOPPING_PATIENCE="3"
WANDB_PROJECT="bert-centralized"

ARGS=(
    "centralized_learning/train.py"
    "--task" "${TASK}"
    "--model-name" "${MODEL_NAME}"
    "--max-seq-length" "${MAX_SEQ_LENGTH}"
    "--lora-r" "${LORA_R}"
    "--lora-alpha" "${LORA_ALPHA}"
    "--lora-target-modules" "${LORA_TARGET_MODULES}"
    "--lora-dropout" "${LORA_DROPOUT}"
    "--epochs" "${EPOCHS}"
    "--learning-rate" "${LEARNING_RATE}"
    "--batch-size" "${BATCH_SIZE}"
    "--grad-accum-steps" "${GRAD_ACCUM_STEPS}"
    "--weight-decay" "${WEIGHT_DECAY}"
    "--lr-scheduler-type" "${LR_SCHEDULER_TYPE}"
    "--seed" "${SEED}"
    "--early-stopping-patience" "${EARLY_STOPPING_PATIENCE}"
)

if [ "${WANDB_ENABLED}" = "true" ] || [ "${WANDB_ENABLED}" = "1" ] || [ "${WANDB_ENABLED}" = "yes" ]; then
    ARGS+=("--wandb" "--wandb-project" "${WANDB_PROJECT}")
fi

echo "=========================================="
echo "Centralized 20 Newsgroups LoRA Training"
echo "  Task                : ${TASK}"
echo "  Model               : ${MODEL_NAME}"
echo "  Epochs              : ${EPOCHS}"
echo "  Batch size          : ${BATCH_SIZE}"
echo "  Grad accum steps    : ${GRAD_ACCUM_STEPS}"
echo "  Max sequence length : ${MAX_SEQ_LENGTH}"
echo "  Seed                : ${SEED}"
echo "  Learning rate       : ${LEARNING_RATE}"
echo "  LR scheduler        : ${LR_SCHEDULER_TYPE}"
echo "  LoRA r              : ${LORA_R}"
echo "  LoRA alpha          : ${LORA_ALPHA}"
echo "  LoRA dropout        : ${LORA_DROPOUT}"
echo "  LoRA targets        : ${LORA_TARGET_MODULES}"
echo "  W&B                 : ${WANDB_ENABLED}"
echo "=========================================="
echo

python "${ARGS[@]}"

echo
echo "=========================================="
echo "20 Newsgroups centralized run complete."
echo "Logs: centralized_learning/logs/"
echo "Checkpoints: centralized_learning/checkpoints/${TASK}_r${LORA_R}/"
echo "=========================================="
