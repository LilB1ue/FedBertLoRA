#!/bin/bash
# Bit-exact verification for `refactor/strategy-utils`.
#
# Runs a short FL simulation on the baseline branch (pre-refactor) and on
# the verify branch (post-refactor) using identical config + seed, then
# diffs `eval_metrics.tsv` (and `clustering.jsonl` when emitted).
# A pure refactor of deterministic functions should produce byte-identical
# logs — any diff means the refactor introduced a behavioural change.
#
# Usage:
#   bash tools/verify_refactor_bit_exact.sh [strategy] [conda-env] [federation]
#
# Examples:
#   bash tools/verify_refactor_bit_exact.sh fedsa
#   bash tools/verify_refactor_bit_exact.sh fedalc-ap-multi exp-flower-bert localhost-gpu
#
# Defaults:
#   strategy   = fedsa            # exercises b_extra_keys path
#   conda-env  = exp-flower-bert
#   federation = local-simulation
#
# Note:
#   - Run from the bert/ project root.
#   - Working tree must be clean (no uncommitted tracked changes).
#   - Untracked files are fine; they survive `git checkout`.
#   - bash holds the script in memory, so the temporary disappearance of
#     this file when checking out the baseline branch is harmless.

set -euo pipefail

STRATEGY="${1:-fedsa}"
CONDA_ENV="${2:-exp-flower-bert}"
FEDERATION="${3:-local-simulation}"

BASELINE_REF="fedalc-agglo-lwc"
VERIFY_REF="refactor/strategy-utils"
TASK="sst2"
ALPHA="0.5"
ROUNDS="3"

TS_BASELINE="REFACTOR_BL_$(date +%s)"
TS_VERIFY="REFACTOR_VR_$(date +%s)"
TMP_DIR=$(mktemp -d)

# ── Pre-flight ─────────────────────────────────────────────────────────────

if ! git diff --quiet HEAD 2>/dev/null; then
    echo "ABORT: working tree has uncommitted tracked changes."
    echo "       Stash or commit first, then re-run."
    git status -sb
    exit 2
fi

ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[info] starting on branch: ${ORIGINAL_BRANCH}"

# Restore branch on any exit (success or failure)
cleanup() {
    local rc=$?
    echo "[info] cleanup: switching back to ${VERIFY_REF}"
    git checkout "${VERIFY_REF}" >/dev/null 2>&1 || true
    rm -rf "${TMP_DIR}"
    exit $rc
}
trap cleanup EXIT

# ── Helpers ────────────────────────────────────────────────────────────────

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

# ── Baseline run (pre-refactor) ────────────────────────────────────────────

git checkout "${BASELINE_REF}"
run_flwr "${TS_BASELINE}" "baseline"

BASELINE_DIR=$(run_dir "${TS_BASELINE}")
BASELINE_TSV="${BASELINE_DIR}/eval_metrics.tsv"
BASELINE_JSONL="${BASELINE_DIR}/clustering.jsonl"

if [[ ! -f "${BASELINE_TSV}" ]]; then
    echo "FAIL: baseline eval_metrics.tsv not found at ${BASELINE_TSV}"
    exit 1
fi
cp "${BASELINE_TSV}" "${TMP_DIR}/baseline_eval.tsv"
if [[ -f "${BASELINE_JSONL}" ]]; then
    cp "${BASELINE_JSONL}" "${TMP_DIR}/baseline_clustering.jsonl"
fi

# ── Verify run (post-refactor) ─────────────────────────────────────────────

git checkout "${VERIFY_REF}"
run_flwr "${TS_VERIFY}" "verify"

VERIFY_DIR=$(run_dir "${TS_VERIFY}")
VERIFY_TSV="${VERIFY_DIR}/eval_metrics.tsv"
VERIFY_JSONL="${VERIFY_DIR}/clustering.jsonl"

if [[ ! -f "${VERIFY_TSV}" ]]; then
    echo "FAIL: verify eval_metrics.tsv not found at ${VERIFY_TSV}"
    exit 1
fi

# ── Diff ───────────────────────────────────────────────────────────────────

echo
echo "=========================================="
echo "Bit-exact verification"
echo "  baseline: ${BASELINE_REF} -> ${TS_BASELINE}"
echo "  verify  : ${VERIFY_REF} -> ${TS_VERIFY}"
echo "=========================================="

PASS=1

echo "[1] eval_metrics.tsv"
if diff -q "${TMP_DIR}/baseline_eval.tsv" "${VERIFY_TSV}" >/dev/null; then
    echo "    PASS: byte-identical"
else
    echo "    FAIL: differs"
    diff -u "${TMP_DIR}/baseline_eval.tsv" "${VERIFY_TSV}" | head -60
    PASS=0
fi

if [[ -f "${TMP_DIR}/baseline_clustering.jsonl" ]] && [[ -f "${VERIFY_JSONL}" ]]; then
    echo "[2] clustering.jsonl"
    if diff -q "${TMP_DIR}/baseline_clustering.jsonl" "${VERIFY_JSONL}" >/dev/null; then
        echo "    PASS: byte-identical"
    else
        echo "    FAIL: differs"
        diff -u "${TMP_DIR}/baseline_clustering.jsonl" "${VERIFY_JSONL}" | head -60
        PASS=0
    fi
fi

echo "=========================================="
if [[ ${PASS} -eq 1 ]]; then
    echo "OVERALL: PASS — refactor preserves bit-exact behaviour"
else
    echo "OVERALL: FAIL — refactor introduced a behavioural change"
fi
echo "=========================================="

exit $((1 - PASS))
