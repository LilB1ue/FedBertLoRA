#!/bin/bash
# Run FedALC-AP-LWC on SST-2 + QNLI with wandb logging
# Usage: bash run_fedalc_ap_lwc.sh [federation] [rounds] [alpha]
#   e.g. bash run_fedalc_ap_lwc.sh                          # default: local-simulation, 30 rounds, α=0.5
#        bash run_fedalc_ap_lwc.sh local-simulation 30 0.3

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-20}"
ALPHA="${3:-0.5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='fedalc-ap-lwc' num-server-rounds=${ROUNDS} dirichlet-alpha=${ALPHA} wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds: ${ROUNDS}, alpha=${ALPHA}, mode=fedalc-ap-lwc"

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
