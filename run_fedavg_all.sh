#!/bin/bash
# Run FedAvg on all GLUE tasks sequentially with wandb logging
# Usage: bash run_fedavg_all.sh [federation] [rounds]
#   e.g. bash run_fedavg_all.sh                          # default: local-simulation, 20 rounds
#        bash run_fedavg_all.sh local-simulation 20

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-20}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='fedavg' num-server-rounds=${ROUNDS} wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds: ${ROUNDS}"

for TASK in sst2 qnli mnli qqp; do
    echo "=========================================="
    echo "Running: ${TASK} (${FEDERATION})"
    echo "=========================================="
    flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"
    echo ""
    echo "${TASK} done."
    echo ""
done

echo "All tasks complete."
