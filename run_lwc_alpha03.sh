#!/bin/bash
# Run FedALC-AP-LWC on SST-2 + QNLI with α=0.3 (fills the missing α=0.3 LWC data).
# Usage: bash run_lwc_alpha03.sh [federation] [rounds]
#   e.g. bash run_lwc_alpha03.sh                     # default: local-simulation, 30 rounds
#        bash run_lwc_alpha03.sh localhost-gpu 20

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-20}"
ALPHA=0.3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='fedalc-ap-lwc' num-server-rounds=${ROUNDS} dirichlet-alpha=${ALPHA} wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds:        ${ROUNDS}"
echo "Alpha:         ${ALPHA}"
echo "Federation:    ${FEDERATION}"
echo "Telemetry:     frozen_layer_indices echoed to clustering.jsonl"

for TASK in sst2 qnli; do
    echo "=========================================="
    echo "Running: ${TASK} FedALC-AP-LWC α=${ALPHA} (${FEDERATION})"
    echo "=========================================="
    flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"
    echo ""
    echo "${TASK} alpha=${ALPHA} done."
    echo ""
done

echo "All tasks complete."
