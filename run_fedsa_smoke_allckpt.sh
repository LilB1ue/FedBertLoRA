#!/bin/bash
# Smoke test: FedSA with the paper-aligned protocol and legacy all-round checkpoint saving.
#
# Usage:
#   bash run_fedsa_smoke_allckpt.sh
#   bash run_fedsa_smoke_allckpt.sh local-simulation sst2 5
#   bash run_fedsa_smoke_allckpt.sh localhost-gpu qnli 5
#
# Args:
#   $1 federation  default: local-simulation
#   $2 task        default: sst2
#   $3 rounds      default: 5

set -euo pipefail

FEDERATION="${1:-local-simulation}"
TASK="${2:-sst2}"
ROUNDS="${3:-5}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)_smoke_fedsa_allckpt"

RUN_CONFIG="\
task-name='${TASK}' \
aggregation-mode='fedsa' \
num-server-rounds=${ROUNDS} \
learning-rate=0.0001 \
lr-schedule='constant' \
lr-scheduler-type='cosine' \
batch-size=32 \
grad-accum-steps=1 \
local-epochs=1 \
dirichlet-alpha=0.5 \
min-partition-size=128 \
test-split-ratio=0.2 \
checkpoint-save-policy='all' \
wandb-enabled=false \
log-timestamp='${TIMESTAMP}'"

echo "=========================================="
echo "FedSA smoke test with all-round checkpoints"
echo "  conda env  : use currently activated environment"
echo "  federation : ${FEDERATION}"
echo "  task       : ${TASK}"
echo "  rounds     : ${ROUNDS}"
echo "  alpha      : 0.5"
echo "  min size   : 128"
echo "  checkpoints: all rounds"
echo "  timestamp  : ${TIMESTAMP}"
echo "=========================================="
echo ""
echo "Logs will be written under:"
echo "  logs/${TIMESTAMP}_fedsa_a0.5/${TASK}_fedsa_a0.5/"
echo ""

flwr run . "${FEDERATION}" --run-config "${RUN_CONFIG}"
