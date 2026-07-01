#!/bin/bash
# Run 20 Newsgroups with Dirichlet alpha=0.3 and the verified partition size.
#
# Usage: bash run_20newsgroups_alpha03.sh [federation] [rounds] [mode] [max_seq_length] [seed] [warmup_rounds] [layer_overlap_trigger] [lora_r] [lora_alpha] [lora_dropout] [lora_target_modules]
#   e.g. bash run_20newsgroups_alpha03.sh
#        bash run_20newsgroups_alpha03.sh localhost-gpu 30 fedalc-agglo-lwc 256 42 5 7 8 16 0.1 query,key,value,dense

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
MODE="${3:-fedalc-agglo-lwc}"
MAX_SEQ_LENGTH="${4:-256}"
SEED="${5:-42}"
WARMUP_ROUNDS="${6:-5}"
LAYER_OVERLAP_TRIGGER="${7:-7}"
LORA_R="${8:-8}"
LORA_ALPHA="${9:-16}"
LORA_DROPOUT="${10:-0.1}"
LORA_TARGET_MODULES="${11:-query,key,value,dense}"
TASK="20newsgroups"
ALPHA="0.3"
MIN_PARTITION_SIZE="128"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CONFIG=(
    "task-name='${TASK}'"
    "aggregation-mode='${MODE}'"
    "num-server-rounds=${ROUNDS}"
    "dirichlet-alpha=${ALPHA}"
    "min-partition-size=${MIN_PARTITION_SIZE}"
    "max-seq-length=${MAX_SEQ_LENGTH}"
    "seed=${SEED}"
    "lora-r=${LORA_R}"
    "lora-alpha=${LORA_ALPHA}"
    "lora-dropout=${LORA_DROPOUT}"
    "lora-target-modules='${LORA_TARGET_MODULES}'"
    "warmup-rounds=${WARMUP_ROUNDS}"
    "layer-overlap-trigger=${LAYER_OVERLAP_TRIGGER}"
    "wandb-enabled=true"
    "log-timestamp='${TIMESTAMP}'"
)
RUN_CONFIG="${CONFIG[*]}"

echo "=========================================="
echo "20 Newsgroups FL run"
echo "  Federation          : ${FEDERATION}"
echo "  Mode                : ${MODE}"
echo "  Rounds              : ${ROUNDS}"
echo "  Alpha               : ${ALPHA}"
echo "  Min partition size  : ${MIN_PARTITION_SIZE}"
echo "  Max sequence length : ${MAX_SEQ_LENGTH}"
echo "  Seed                : ${SEED}"
echo "  LoRA r              : ${LORA_R}"
echo "  LoRA alpha          : ${LORA_ALPHA}"
echo "  LoRA dropout        : ${LORA_DROPOUT}"
echo "  LoRA targets        : ${LORA_TARGET_MODULES}"
echo "  Warmup rounds       : ${WARMUP_ROUNDS}"
echo "  Layer overlap trig. : ${LAYER_OVERLAP_TRIGGER}"
echo "  W&B                 : enabled"
echo "  Timestamp           : ${TIMESTAMP}"
echo "=========================================="
echo

flwr run . "${FEDERATION}" --run-config "${RUN_CONFIG}"
STATUS=$?

echo
echo "=========================================="
if [ ${STATUS} -ne 0 ]; then
    echo "FAIL: 20 Newsgroups run exited with status ${STATUS}"
    exit ${STATUS}
fi

echo "20 Newsgroups run complete."
