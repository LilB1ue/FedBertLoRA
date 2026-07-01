#!/bin/bash
# Run only the 20 Newsgroups FFA-LoRA baseline.
#
# Usage: bash run_20newsgroups_ffa_alpha05.sh [federation] [rounds] [max_seq_length] [seed] [learning_rate] [lr_schedule] [batch_size] [lora_r] [lora_alpha] [lora_dropout] [lora_target_modules] [wandb_enabled] [timestamp]
#   e.g. bash run_20newsgroups_ffa_alpha05.sh
#        bash run_20newsgroups_ffa_alpha05.sh local-simulation 30 256 42 0.0001 constant 16 8 16 0.1 query,key,value,dense true 20260624_115537

set -euo pipefail

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
MAX_SEQ_LENGTH="${3:-256}"
SEED="${4:-42}"
LEARNING_RATE="${5:-0.0001}"
LR_SCHEDULE="${6:-constant}"
BATCH_SIZE="${7:-16}"
LORA_R="${8:-8}"
LORA_ALPHA="${9:-16}"
LORA_DROPOUT="${10:-0.1}"
LORA_TARGET_MODULES="${11:-query,key,value,dense}"
WANDB_ENABLED_RAW="${12:-true}"
TIMESTAMP="${13:-$(date +%Y%m%d_%H%M%S)}"

TASK="20newsgroups"
MODE="ffa"
ALPHA="0.5"
MIN_PARTITION_SIZE="128"

case "${WANDB_ENABLED_RAW}" in
    true|True|TRUE|1|yes|Yes|YES)
        WANDB_ENABLED="true"
        ;;
    *)
        WANDB_ENABLED="false"
        ;;
esac

CONFIG=(
    "task-name='${TASK}'"
    "aggregation-mode='${MODE}'"
    "num-server-rounds=${ROUNDS}"
    "dirichlet-alpha=${ALPHA}"
    "min-partition-size=${MIN_PARTITION_SIZE}"
    "max-seq-length=${MAX_SEQ_LENGTH}"
    "seed=${SEED}"
    "learning-rate=${LEARNING_RATE}"
    "lr-schedule='${LR_SCHEDULE}'"
    "batch-size=${BATCH_SIZE}"
    "lora-r=${LORA_R}"
    "lora-alpha=${LORA_ALPHA}"
    "lora-dropout=${LORA_DROPOUT}"
    "lora-target-modules='${LORA_TARGET_MODULES}'"
    "checkpoint-save-policy='selective'"
    "wandb-enabled=${WANDB_ENABLED}"
    "log-timestamp='${TIMESTAMP}'"
)
RUN_CONFIG="${CONFIG[*]}"
LOG_PATH="logs/${TIMESTAMP}_${MODE}_a${ALPHA}/${TASK}_${MODE}_a${ALPHA}"

echo "=========================================="
echo "20 Newsgroups FFA-LoRA baseline run"
echo "  Federation          : ${FEDERATION}"
echo "  Mode                : ${MODE}"
echo "  Rounds              : ${ROUNDS}"
echo "  Alpha               : ${ALPHA}"
echo "  Min partition size  : ${MIN_PARTITION_SIZE}"
echo "  Max sequence length : ${MAX_SEQ_LENGTH}"
echo "  Seed                : ${SEED}"
echo "  Learning rate       : ${LEARNING_RATE}"
echo "  LR schedule         : ${LR_SCHEDULE}"
echo "  Batch size          : ${BATCH_SIZE}"
echo "  LoRA r              : ${LORA_R}"
echo "  LoRA alpha          : ${LORA_ALPHA}"
echo "  LoRA dropout        : ${LORA_DROPOUT}"
echo "  LoRA targets        : ${LORA_TARGET_MODULES}"
echo "  Checkpoints         : selective"
echo "  W&B                 : ${WANDB_ENABLED}"
echo "  Timestamp           : ${TIMESTAMP}"
echo "=========================================="
echo
echo "Expected outputs:"
echo "  ${LOG_PATH}/fit_metrics.tsv"
echo "  ${LOG_PATH}/eval_metrics.tsv"
echo "  ${LOG_PATH}/server_eval.tsv"
echo "  ${LOG_PATH}/best_checkpoints/best_round.json"
echo

flwr run . "${FEDERATION}" --run-config "${RUN_CONFIG}"
STATUS=$?

echo
echo "=========================================="
if [ ${STATUS} -ne 0 ]; then
    echo "FAIL: ${TASK} ${MODE} exited with status ${STATUS}"
    exit ${STATUS}
fi

echo "20 Newsgroups FFA-LoRA run complete."
echo "Logs: ${LOG_PATH}/"
