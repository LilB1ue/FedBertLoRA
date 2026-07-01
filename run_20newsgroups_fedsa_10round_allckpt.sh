#!/bin/bash
# Run FedSA-LoRA on 20 Newsgroups with all-round checkpoints for analysis.
#
# This is intended for post-hoc B-representation/layerwise clustering analysis:
# every round's post-training client checkpoints are retained under
# client_checkpoints/round_R/client_ID.
#
# Usage:
#   bash run_20newsgroups_fedsa_10round_allckpt.sh
#   bash run_20newsgroups_fedsa_10round_allckpt.sh localhost-gpu
#   bash run_20newsgroups_fedsa_10round_allckpt.sh localhost-gpu 10 256 42
#
# Args:
#   $1 federation      default: local-simulation
#   $2 rounds          default: 10
#   $3 max_seq_length  default: 256
#   $4 seed            default: 42
#   $5 learning_rate   default: 0.0001
#   $6 lr_schedule     default: constant
#   $7 batch_size      default: 16
#   $8 wandb_enabled   default: false

set -euo pipefail

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-10}"
MAX_SEQ_LENGTH="${3:-256}"
SEED="${4:-42}"
LEARNING_RATE="${5:-0.0001}"
LR_SCHEDULE="${6:-constant}"
BATCH_SIZE="${7:-16}"
WANDB_ENABLED="${8:-false}"

TASK="20newsgroups"
MODE="fedsa"
ALPHA="0.5"
MIN_PARTITION_SIZE="128"
NUM_CLIENTS="30"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.1"
LORA_TARGET_MODULES="query,key,value,dense"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)_20news_fedsa_10round_allckpt"

CONFIG=(
    "task-name='${TASK}'"
    "aggregation-mode='${MODE}'"
    "num-server-rounds=${ROUNDS}"
    "num-clients=${NUM_CLIENTS}"
    "fraction-fit=1.0"
    "dirichlet-alpha=${ALPHA}"
    "min-partition-size=${MIN_PARTITION_SIZE}"
    "max-seq-length=${MAX_SEQ_LENGTH}"
    "seed=${SEED}"
    "learning-rate=${LEARNING_RATE}"
    "lr-schedule='${LR_SCHEDULE}'"
    "batch-size=${BATCH_SIZE}"
    "grad-accum-steps=1"
    "local-epochs=1"
    "test-split-ratio=0.2"
    "lora-r=${LORA_R}"
    "lora-alpha=${LORA_ALPHA}"
    "lora-dropout=${LORA_DROPOUT}"
    "lora-target-modules='${LORA_TARGET_MODULES}'"
    "checkpoint-save-policy='all'"
    "wandb-enabled=${WANDB_ENABLED}"
    "wandb-run-tag='20newsgroups_fedsa_10round_allckpt'"
    "log-timestamp='${TIMESTAMP}'"
)
RUN_CONFIG="${CONFIG[*]}"

LOG_PATH="logs/${TIMESTAMP}_${MODE}_a${ALPHA}/${TASK}_${MODE}_a${ALPHA}"

echo "=========================================="
echo "20 Newsgroups FedSA all-checkpoint run"
echo "  Federation          : ${FEDERATION}"
echo "  Mode                : ${MODE}"
echo "  Rounds              : ${ROUNDS}"
echo "  Alpha               : ${ALPHA}"
echo "  Min partition size  : ${MIN_PARTITION_SIZE}"
echo "  Num clients         : ${NUM_CLIENTS}"
echo "  Max sequence length : ${MAX_SEQ_LENGTH}"
echo "  Seed                : ${SEED}"
echo "  Learning rate       : ${LEARNING_RATE}"
echo "  LR schedule         : ${LR_SCHEDULE}"
echo "  Batch size          : ${BATCH_SIZE}"
echo "  LoRA r              : ${LORA_R}"
echo "  LoRA alpha          : ${LORA_ALPHA}"
echo "  LoRA dropout        : ${LORA_DROPOUT}"
echo "  LoRA targets        : ${LORA_TARGET_MODULES}"
echo "  Checkpoints         : all rounds"
echo "  W&B                 : ${WANDB_ENABLED}"
echo "  Timestamp           : ${TIMESTAMP}"
echo "=========================================="
echo
echo "Expected outputs:"
echo "  ${LOG_PATH}/fit_metrics.tsv"
echo "  ${LOG_PATH}/eval_metrics.tsv"
echo "  ${LOG_PATH}/client_checkpoints/round_R/client_ID/"
echo "  ${LOG_PATH}/received_checkpoints/round_R/client_ID/"
echo

flwr run . "${FEDERATION}" --run-config "${RUN_CONFIG}"
STATUS=$?

echo
echo "=========================================="
if [ ${STATUS} -ne 0 ]; then
    echo "FAIL: 20 Newsgroups FedSA run exited with status ${STATUS}"
    exit ${STATUS}
fi

echo "20 Newsgroups FedSA all-checkpoint run complete."
echo "Logs: ${LOG_PATH}/"
