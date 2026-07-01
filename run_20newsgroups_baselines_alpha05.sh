#!/bin/bash
# Run 20 Newsgroups baselines: FedAvg, FedSA-LoRA, and FFA-LoRA.
#
# Usage: bash run_20newsgroups_baselines_alpha05.sh [federation] [rounds] [max_seq_length] [seed] [learning_rate] [lr_schedule] [batch_size] [lora_r] [lora_alpha] [lora_dropout] [lora_target_modules]
#   e.g. bash run_20newsgroups_baselines_alpha05.sh
#        bash run_20newsgroups_baselines_alpha05.sh localhost-gpu 30 256 42 0.0001 constant 16 8 16 0.1 query,key,value,dense

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
TASK="20newsgroups"
ALPHA="0.5"
MIN_PARTITION_SIZE="128"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

COMMON=(
    "task-name='${TASK}'"
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
    "wandb-enabled=true"
    "log-timestamp='${TIMESTAMP}'"
)
COMMON_CONFIG="${COMMON[*]}"

echo "=========================================="
echo "20 Newsgroups baseline runs"
echo "  Federation          : ${FEDERATION}"
echo "  Modes               : fedavg fedsa ffa"
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
echo "  W&B                 : enabled"
echo "  Timestamp           : ${TIMESTAMP}"
echo "=========================================="
echo

for MODE in fedavg fedsa ffa; do
    echo "=========================================="
    echo "Running: ${TASK} ${MODE} (${FEDERATION})"
    echo "=========================================="
    flwr run . "${FEDERATION}" --run-config "aggregation-mode='${MODE}' ${COMMON_CONFIG}"
    STATUS=$?

    echo
    if [ ${STATUS} -ne 0 ]; then
        echo "FAIL: ${TASK} ${MODE} exited with status ${STATUS}"
        exit ${STATUS}
    fi
    echo "${TASK} ${MODE} done."
    echo
done

echo "All 20 Newsgroups baseline runs complete."
