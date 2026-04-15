#!/bin/bash
# Run FedALC-LoRA with alpha=0.3 on SST-2 + QNLI
# Usage: bash run_fedalc_alpha03.sh [federation] [rounds]
# Note: alpha=0.1 fails with DirichletPartitioner (30 clients + binary task → empty partitions)

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='fedalc' num-server-rounds=${ROUNDS} dirichlet-alpha=0.3 wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds: ${ROUNDS}, alpha=0.3"

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
