#!/bin/bash
# Apply the rename manifest produced by scan_log_inventory.py.
# Dry-run by default; pass --apply to actually perform mv.
#
# Usage:
#   bash tools/apply_log_rename.sh            # dry run
#   bash tools/apply_log_rename.sh --apply    # perform renames

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${REPO}/tools/log_rename_manifest.tsv"
APPLY=0

if [[ "${1:-}" == "--apply" ]]; then
    APPLY=1
    echo "=== APPLY mode: renames will be executed ==="
else
    echo "=== DRY-RUN mode (pass --apply to execute) ==="
fi

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Manifest not found: ${MANIFEST}"
    echo "Run: python3 tools/scan_log_inventory.py"
    exit 1
fi

n_apply=0
n_skip=0
n_missing=0

# Skip header line
tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r old_path new_path alpha source note; do
    if [[ "${new_path}" == "SKIP" ]]; then
        echo "SKIP (unknown alpha): ${old_path}  -- ${note}"
        ((n_skip++)) || true
        continue
    fi
    src="${REPO}/${old_path}"
    dst="${REPO}/${new_path}"
    if [[ ! -e "${src}" ]]; then
        echo "MISSING ${src}"
        ((n_missing++)) || true
        continue
    fi
    if [[ -e "${dst}" ]]; then
        echo "ALREADY RENAMED: ${dst}"
        ((n_skip++)) || true
        continue
    fi
    echo "  ${old_path}  ->  ${new_path}  [alpha=${alpha}]"
    if [[ -n "${note}" ]]; then
        echo "      NOTE: ${note}"
    fi
    if [[ ${APPLY} -eq 1 ]]; then
        mv "${src}" "${dst}"
    fi
    ((n_apply++)) || true
done

echo ""
echo "Summary:"
echo "  renames (dry=$([ ${APPLY} -eq 0 ] && echo 'pending' || echo 'applied')): ${n_apply}"
echo "  skipped: ${n_skip}"
echo "  missing: ${n_missing}"
