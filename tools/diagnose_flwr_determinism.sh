#!/bin/bash
# Diagnose whether flwr local-simulation is deterministic across runs.
#
# Runs the SAME config TWICE on the SAME branch (no checkout, no refactor
# involved). If the two outputs differ → flwr simulation has non-determinism,
# in which case bit-exact verification across branches is invalid.
#
# Expected outcome: outputs should be byte-identical. If they're not, the
# pattern (R1 same, R2+ diverges) tells us what kind of non-determinism it is.
#
# Usage: bash tools/diagnose_flwr_determinism.sh [strategy] [conda-env]

set -euo pipefail

STRATEGY="${1:-fedsa}"
CONDA_ENV="${2:-exp-flower-bert}"
FEDERATION="local-simulation"

TASK="sst2"
ALPHA="0.5"
ROUNDS="3"

TS_A="DETERM_A_$(date +%s)"
sleep 1
TS_B="DETERM_B_$(date +%s)"

run_flwr() {
    local timestamp="$1"
    local label="$2"
    echo
    echo "=========================================="
    echo "[${label}] flwr run on ${STRATEGY}, ${TASK}, alpha=${ALPHA}, rounds=${ROUNDS}"
    echo "  log-timestamp = ${timestamp}"
    echo "=========================================="
    conda run -n "${CONDA_ENV}" --no-capture-output \
        flwr run . "${FEDERATION}" --run-config \
        "task-name='${TASK}' aggregation-mode='${STRATEGY}' num-server-rounds=${ROUNDS} dirichlet-alpha=${ALPHA} log-timestamp='${timestamp}' wandb-enabled=false"
}

run_dir() {
    local ts="$1"
    echo "logs/${ts}_${STRATEGY}_a${ALPHA}/${TASK}_${STRATEGY}_a${ALPHA}"
}

run_flwr "${TS_A}" "run A"
run_flwr "${TS_B}" "run B"

A_TSV="$(run_dir "${TS_A}")/eval_metrics.tsv"
B_TSV="$(run_dir "${TS_B}")/eval_metrics.tsv"

echo
echo "=========================================="
echo "Determinism diagnostic"
echo "  Same branch ($(git rev-parse --abbrev-ref HEAD)), same config, two runs."
echo "=========================================="

if diff -q "${A_TSV}" "${B_TSV}" >/dev/null; then
    echo "RESULT: byte-identical → flwr IS deterministic"
    echo "  Implication: previous bit-exact FAIL means refactor has a real bug."
else
    echo "RESULT: differs → flwr is NOT deterministic"
    echo "  Implication: previous bit-exact FAIL was likely a false positive"
    echo "  (refactor may still be correct)."
    echo
    echo "First 60 lines of diff:"
    diff -u "${A_TSV}" "${B_TSV}" | head -60
fi
