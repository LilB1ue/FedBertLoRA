#!/bin/bash
# Run FFA-LoRA on SST-2 + QNLI with wandb logging
# Usage: bash run_ffa_all.sh [federation] [rounds]

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds: ${ROUNDS}"

for ALPHA in 0.5 0.3; do
    for TASK in sst2 qnli; do
        COMMON="aggregation-mode='ffa' num-server-rounds=${ROUNDS} dirichlet-alpha=${ALPHA} wandb-enabled=true log-timestamp='${TIMESTAMP}'"
        echo "=========================================="
        echo "Running: ${TASK} FFA-LoRA alpha=${ALPHA} (${FEDERATION})"
        echo "=========================================="
        flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"
        echo ""
        echo "${TASK} alpha=${ALPHA} done."
        echo ""
    done
done

echo "All tasks complete."
