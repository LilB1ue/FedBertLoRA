#!/bin/bash
# Run FedALC-Random (control baseline for FedALC-AP) on GLUE tasks.
#
# Purpose: ablate the AP clustering signal — Random keeps the per-cluster B
# aggregation mechanism but assigns clients to clusters at random. If FedALC-AP
# beats FedALC-Random, the AP similarity-based grouping is contributing; if
# they tie, the win comes from the cluster-bucket scheme alone (or just from
# keeping B partly local).
#
# Usage:
#   bash run_fedalc_random.sh                                       # defaults: SST-2 + QNLI, α=0.5, 30r, K=3
#   bash run_fedalc_random.sh local-simulation 30 0.3               # α=0.3
#   bash run_fedalc_random.sh local-simulation 30 0.5 5             # K=5
#   bash run_fedalc_random.sh local-simulation 30 0.5 3 false       # re-draw clusters every round
#
# Args:
#   $1 federation       (default: local-simulation)
#   $2 rounds           (default: 30)
#   $3 dirichlet-alpha  (default: 0.5)
#   $4 random-cluster-k (default: 3)  — typically set to AP's mean K under the same α
#   $5 fixed-assignment (default: true) — true=round-1 partition reused; false=re-draw every round
#   $6 random-seed      (default: 42)

set -euo pipefail

FEDERATION="${1:-local-simulation}"
ROUNDS="${2:-30}"
ALPHA="${3:-0.5}"
RAND_K="${4:-3}"
FIXED="${5:-true}"
SEED="${6:-42}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

COMMON="aggregation-mode='fedalc-random' \
num-server-rounds=${ROUNDS} \
dirichlet-alpha=${ALPHA} \
random-cluster-k=${RAND_K} \
random-fixed-assignment=${FIXED} \
random-seed=${SEED} \
wandb-enabled=true \
log-timestamp='${TIMESTAMP}'"

echo "=========================================="
echo "FedALC-Random control baseline"
echo "  federation : ${FEDERATION}"
echo "  rounds     : ${ROUNDS}"
echo "  alpha      : ${ALPHA}"
echo "  K          : ${RAND_K}"
echo "  fixed      : ${FIXED}"
echo "  seed       : ${SEED}"
echo "  timestamp  : ${TIMESTAMP}"
echo "=========================================="

for TASK in sst2 qnli; do
    echo ""
    echo "--- Running: ${TASK} ---"
    flwr run . "${FEDERATION}" --run-config "task-name='${TASK}' ${COMMON}"
    echo "${TASK} done."
done

echo ""
echo "All tasks complete. Logs at: logs/${TIMESTAMP}_fedalc-random_a${ALPHA}/"
