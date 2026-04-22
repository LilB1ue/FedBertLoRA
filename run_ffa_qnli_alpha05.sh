#!/bin/bash
# Run FFA-LoRA on QNLI only with alpha=0.5 and wandb logging.
# Fills the missing QNLI α=0.5 client-side run (plot_method_comparison marks it server-only).
# Usage: bash run_ffa_qnli_alpha05.sh [federation] [rounds]
#   e.g. bash run_ffa_qnli_alpha05.sh                 # default: local-simulation, 30 rounds
#        bash run_ffa_qnli_alpha05.sh localhost-gpu 30

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
ALPHA=0.5
TASK=qnli
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Run timestamp: ${TIMESTAMP}"
echo "Rounds:        ${ROUNDS}"
echo "Alpha:         ${ALPHA}"
echo "Task:          ${TASK}"
echo "Federation:    ${FEDERATION}"

COMMON="aggregation-mode='ffa' num-server-rounds=${ROUNDS} dirichlet-alpha=${ALPHA} wandb-enabled=true log-timestamp='${TIMESTAMP}'"

echo "=========================================="
echo "Running: ${TASK} FFA-LoRA alpha=${ALPHA} (${FEDERATION})"
echo "=========================================="
flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"

echo ""
echo "${TASK} alpha=${ALPHA} done."
