#!/bin/bash
# Second-pass rename: move logs/{ts}/{task}_{mode}_a{alpha}/ subdirs under a
# new parent logs/{ts}_{mode}_a{alpha}/ (batch dir). Keeps inner subdir name
# intact so all existing tools that accept individual run paths keep working.
#
# Usage:
#   bash tools/apply_batch_dir_rename.sh             # dry run
#   bash tools/apply_batch_dir_rename.sh --apply     # perform renames

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="${REPO}/logs"
APPLY=0

if [[ "${1:-}" == "--apply" ]]; then
    APPLY=1
    echo "=== APPLY mode ==="
else
    echo "=== DRY-RUN mode (pass --apply to execute) ==="
fi

mapfile -t TS_DIRS < <(find "${LOGS}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)

n_moved=0
n_skipped=0
n_kept=0
declare -a cleanup_candidates=()

for ts_dir in "${TS_DIRS[@]}"; do
    # Only handle timestamp-shaped top-level dirs (YYYYMMDD_HHMMSS, may have _ai2 etc.)
    if [[ ! "${ts_dir}" =~ ^[0-9]{8}_[0-9]{6} ]]; then
        continue
    fi
    # Skip if this top-level dir is already a batch dir (contains _<mode>_a)
    if [[ "${ts_dir}" =~ _a(0\.[0-9]+|UNKNOWN) ]]; then
        echo "ALREADY BATCH-FORM: ${ts_dir}"
        ((n_skipped++)) || true
        continue
    fi

    ts_path="${LOGS}/${ts_dir}"
    # Find inner subdirs like <task>_<mode>_a<alpha>
    mapfile -t subs < <(find "${ts_path}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
    if [[ ${#subs[@]} -eq 0 ]]; then
        continue
    fi

    moved_any=0
    for sub in "${subs[@]}"; do
        # Expect pattern: task_mode_a<alpha>
        if [[ ! "${sub}" =~ ^([a-z0-9]+)_(.+)_a([0-9.]+|UNKNOWN)$ ]]; then
            echo "  UNMATCHED subdir pattern in ${ts_dir}/${sub} — skipping"
            continue
        fi
        mode="${BASH_REMATCH[2]}"
        alpha="${BASH_REMATCH[3]}"

        new_ts_dir="${ts_dir}_${mode}_a${alpha}"
        new_parent="${LOGS}/${new_ts_dir}"
        old_full="${ts_path}/${sub}"
        new_full="${new_parent}/${sub}"

        if [[ -e "${new_full}" ]]; then
            echo "SKIP (exists): ${new_full}"
            continue
        fi

        echo "  ${ts_dir}/${sub}  ->  ${new_ts_dir}/${sub}"
        if [[ ${APPLY} -eq 1 ]]; then
            mkdir -p "${new_parent}"
            mv "${old_full}" "${new_full}"
        fi
        ((n_moved++)) || true
        moved_any=1
    done

    if [[ ${moved_any} -eq 1 ]]; then
        cleanup_candidates+=("${ts_path}")
    else
        ((n_kept++)) || true
    fi
done

# Remove now-empty top-level timestamp dirs
echo ""
echo "Cleaning up empty parent dirs:"
for cand in "${cleanup_candidates[@]}"; do
    # Only rmdir if empty
    if [[ -d "${cand}" ]]; then
        if [[ -z "$(ls -A "${cand}" 2>/dev/null || true)" ]]; then
            echo "  rmdir ${cand}"
            if [[ ${APPLY} -eq 1 ]]; then
                rmdir "${cand}"
            fi
        else
            echo "  KEEP (not empty): ${cand}"
        fi
    fi
done

echo ""
echo "Summary: moved=${n_moved} kept=${n_kept} skipped=${n_skipped}"
