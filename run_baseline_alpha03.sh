#!/bin/bash
# Run FedAvg + FedSA baselines with alpha=0.3 on SST-2 + QNLI
# Usage: bash run_baseline_alpha03.sh [federation] [rounds]

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds: ${ROUNDS}, alpha=0.3"

for MODE in fedavg fedsa; do
    COMMON="aggregation-mode='${MODE}' num-server-rounds=${ROUNDS} dirichlet-alpha=0.3 wandb-enabled=true log-timestamp='${TIMESTAMP}'"
    for TASK in sst2 qnli; do
        echo "=========================================="
        echo "Running: ${TASK} ${MODE} (${FEDERATION})"
        echo "=========================================="
        flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"
        echo ""
        echo "${TASK} ${MODE} done."
        echo ""
    done
done

echo "All tasks complete."
