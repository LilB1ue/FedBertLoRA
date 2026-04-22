#!/bin/bash
# Smoke: LWC (no warm-up) on QNLI α=0.5, 5 rounds only.
# Purpose: verify singleton pattern + get partition-id-mapped clustering.jsonl.
# Usage: bash run_lwc_qnli_smoke.sh [federation]

FEDERATION="${1:-local-simulation}"
ROUNDS=5
ALPHA=0.5
TASK=qnli
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_SUBDIR="${TIMESTAMP}_fedalc-lwc_a${ALPHA}"

echo "Timestamp: ${TIMESTAMP}"
echo "Rounds:    ${ROUNDS}"
echo "Task:      ${TASK}  α=${ALPHA}  mode=fedalc-ap-lwc"

flwr run . "${FEDERATION}" --run-config "\
task-name='${TASK}' \
aggregation-mode='fedalc-ap-lwc' \
num-server-rounds=${ROUNDS} \
dirichlet-alpha=${ALPHA} \
wandb-enabled=false \
log-timestamp='${TIMESTAMP}'"

echo ""
echo "=== clustering.jsonl (partition_ids should be 0-29 integers) ==="
cat "logs/${LOG_SUBDIR}/${TASK}_fedalc-lwc_a${ALPHA}/clustering.jsonl" | python3 -c "
import json, sys
from collections import Counter
for line in sys.stdin:
    d = json.loads(line)
    sizes = [len(v) for v in d['clusters'].values()]
    singletons = [pids[0] for pids in d['clusters'].values() if len(pids)==1]
    print(f'R{d[\"round\"]:2d}  n_cl={d[\"n_clusters\"]}  sizes={sizes}  sil={d.get(\"silhouette_score\",\"?\"):.3f}  singleton_pids={singletons}')
"
