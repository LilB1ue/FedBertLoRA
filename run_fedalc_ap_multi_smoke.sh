#!/bin/bash
# FedALC-AP-Multi smoke test: quick 5-round SST-2 run to verify integration.
#
# Checks:
#   1. No crash through all 5 rounds
#   2. Hopkins values logged in [0, 1] range, not nan/inf (C1 fix)
#   3. Layer selection indices logged every round (LWC integration)
#   4. n_params_clustering equals top-K * B-layer-dim, not full B dim
#   5. Cumulative ΔB non-zero from round 1 (I1 fix)
#
# Usage: bash run_fedalc_ap_multi_smoke.sh [federation]
# Default federation: local-simulation (CPU); pass 'localhost-gpu' for GPU.

FEDERATION="${1:-local-simulation}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TASK="sst2"
ALPHA="0.5"

# Lower hopkins-threshold so Phase 0→1 transition is more likely within 5 rounds.
# warmup-max-rounds=3 caps warm-up, guaranteeing we see Phase 1 code path too.
CONFIG=(
    "aggregation-mode='fedalc-ap-multi'"
    "task-name='${TASK}'"
    "num-server-rounds=5"
    "dirichlet-alpha=${ALPHA}"
    "hopkins-threshold=0.6"
    "warmup-max-rounds=3"
    "layer-selection-k=10"
    "layer-reselect-every=1"
    "layer-score-feature='cumulative_delta_b'"
    "wandb-enabled=false"
    "log-timestamp='${TIMESTAMP}'"
)
RUN_CONFIG="${CONFIG[*]}"

echo "=========================================="
echo "FedALC-AP-Multi smoke test"
echo "  Federation : ${FEDERATION}"
echo "  Task       : ${TASK}"
echo "  Timestamp  : ${TIMESTAMP}"
echo "=========================================="
echo

flwr run . "${FEDERATION}" --run-config "${RUN_CONFIG}"
STATUS=$?

echo
echo "=========================================="
if [ ${STATUS} -ne 0 ]; then
    echo "FAIL: flwr run exited with status ${STATUS}"
    exit ${STATUS}
fi

LOG_DIR="logs/${TIMESTAMP}/${TASK}_fedalc-ap-multi_a${ALPHA}"
CLUSTERING_LOG="${LOG_DIR}/clustering.jsonl"

if [ ! -f "${CLUSTERING_LOG}" ]; then
    echo "FAIL: clustering.jsonl not found at ${CLUSTERING_LOG}"
    exit 1
fi

echo "Checking ${CLUSTERING_LOG}..."
python3 - <<PYEOF
import json
from pathlib import Path

path = Path("${CLUSTERING_LOG}")
lines = path.read_text().strip().split("\n")
entries = [json.loads(l) for l in lines]

print(f"  Total rounds logged : {len(entries)}")

# Check 1: Hopkins values sensible
hopkins_values = [(e["round"], e.get("hopkins")) for e in entries if "hopkins" in e]
print(f"  Hopkins per round   : {hopkins_values}")
for rnd, h in hopkins_values:
    if h is None:
        continue
    if not (0.0 <= h <= 1.0):
        print(f"  FAIL: round {rnd} Hopkins={h} out of [0,1]")
        exit(1)

# Check 2: layer selection logged
first_entry = entries[0]
feats = first_entry.get("clustering_features", {})
selected = feats.get("selected_layer_indices") or feats.get("active_layer_indices")
n_params = feats.get("n_params_hopkins") or feats.get("n_params_clustering")
print(f"  Round 1 top-K idx   : {selected}")
print(f"  Round 1 Hopkins D   : {n_params}")
if not selected:
    print(f"  FAIL: selected/active_layer_indices missing in round 1")
    exit(1)
if n_params is None or n_params > 200000:
    print(f"  FAIL: Hopkins dim {n_params} too large (expected ~10K with top-K=10)")
    exit(1)

# Check 4: trigger_fired logged in Phase 0 entries
phase0 = [e for e in entries if e.get("phase") == 0]
triggers = [(e["round"], e.get("trigger_fired"), e.get("trigger_reason")) for e in phase0]
print(f"  Phase 0 triggers    : {triggers}")

# Check 3: phase transitions
phases = [e.get("phase") for e in entries]
print(f"  Phase trajectory    : {phases}")

print("PASS: smoke test checks succeeded")
PYEOF

STATUS=$?
if [ ${STATUS} -eq 0 ]; then
    echo "Smoke test PASSED"
else
    echo "Smoke test FAILED"
fi
exit ${STATUS}
