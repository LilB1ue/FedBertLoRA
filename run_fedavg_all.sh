#!/bin/bash｀｀
# Run FedAvg on all GLUE tasks sequentially with wandb logging
# Usage: bash run_fedavg_all.sh [localhost|localhost-gpu]

FEDERATION="${1:-localhost-gpu}"
COMMON="aggregation-mode='fedavg' wandb-enabled=true"

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
