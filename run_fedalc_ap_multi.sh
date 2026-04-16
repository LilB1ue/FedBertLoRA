#!/bin/bash
# Run FedALC-AP-Multi (AP + built-in layer selection + Hopkins adaptive trigger
# + cumulative ΔB + freeze) on SST-2 + QNLI
# Usage: bash run_fedalc_ap_multi.sh [federation] [rounds] [alpha]

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
ALPHA="${3:-0.5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='fedalc-ap-multi' num-server-rounds=${ROUNDS} dirichlet-alpha=${ALPHA} wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds: ${ROUNDS}, alpha=${ALPHA}, mode=fedalc-ap-multi"

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
