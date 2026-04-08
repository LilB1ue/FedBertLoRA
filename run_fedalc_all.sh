#!/bin/bash
# Run FedALC-LoRA on GLUE tasks sequentially with wandb logging
# Usage: bash run_fedalc_all.sh [federation] [rounds]
#   e.g. bash run_fedalc_all.sh                          # default: local-simulation, 30 rounds
#        bash run_fedalc_all.sh local-simulation 50

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='fedalc' num-server-rounds=${ROUNDS} wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds: ${ROUNDS}"

for TASK in sst2 qnli; do
    echo "=========================================="
    echo "Running: ${TASK} (${FEDERATION})"
    echo "=========================================="
    flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"
    echo ""
    echo "${TASK} done."
    echo ""
done

echo "All tasks complete."
