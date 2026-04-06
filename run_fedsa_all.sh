#!/bin/bash
# Run FedSA-LoRA on all GLUE tasks sequentially with wandb logging
# Usage: bash run_fedsa_all.sh [local-simulation]

FEDERATION="${1:-local-simulation}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='fedsa' wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"

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
