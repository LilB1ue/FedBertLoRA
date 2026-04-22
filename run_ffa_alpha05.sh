#!/bin/bash
# Run FFA-LoRA on SST-2 + QNLI with alpha=0.5 and wandb logging
# Usage: bash run_ffa_alpha05.sh [federation] [rounds]
#   e.g. bash run_ffa_alpha05.sh                     # default: local-simulation, 30 rounds
#        bash run_ffa_alpha05.sh localhost-gpu 30
#
# Note: recreates the FFA alpha=0.5 runs that were previously overwritten
# (see commit aa65b63). log_subdir now carries the _a<alpha> suffix so
# concurrent-alpha runs cannot collide again.

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
ALPHA=0.5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON="aggregation-mode='ffa' num-server-rounds=${ROUNDS} dirichlet-alpha=${ALPHA} wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds:        ${ROUNDS}"
echo "Alpha:         ${ALPHA}"
echo "Federation:    ${FEDERATION}"

for TASK in sst2 qnli; do
    echo "=========================================="
    echo "Running: ${TASK} FFA-LoRA alpha=${ALPHA} (${FEDERATION})"
    echo "=========================================="
    flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"
    echo ""
    echo "${TASK} alpha=${ALPHA} done."
    echo ""
done

echo "All tasks complete."
