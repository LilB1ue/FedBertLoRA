"""Build inventory of log dirs and match to wandb runs → true alpha.

wandb config.yaml does NOT record log_timestamp, so we match by the rule:
  each wandb run's start time must be >= its corresponding log_timestamp,
  and closest match for matching (task, mode).

Overwrite detection: when multiple wandb runs (different alphas) map to the
same log subdir, the LAST-STARTED run is the one whose results survive on
disk (matches server_app.py's "w" then "a" write pattern on each new process).

Outputs:
  - tools/log_inventory.tsv: per-wandb-run → log subdir mapping
  - tools/log_rename_manifest.tsv: log subdir → new name with alpha tag
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WANDB_DIR = REPO / "wandb"
LOGS_DIR = REPO / "logs"
OUT_INVENTORY = REPO / "tools" / "log_inventory.tsv"
OUT_MANIFEST = REPO / "tools" / "log_rename_manifest.tsv"

# How far after log_timestamp we still count a wandb run as belonging to it.
MATCH_WINDOW = timedelta(hours=24)


def _scalar(entry):
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def parse_wandb_config(config_path: Path):
    try:
        with config_path.open() as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        return None, f"yaml error: {exc}"

    def get(*names):
        for n in names:
            if n in cfg:
                val = _scalar(cfg[n])
                if val is not None:
                    return val
        return None

    entry = {
        "task": get("task_name"),
        "mode": get("aggregation_mode"),
        "alpha": get("dirichlet_alpha"),
        "rounds": get("num_server_rounds"),
    }
    if any(entry[k] is None for k in ("task", "mode", "alpha")):
        return entry, "missing required keys"
    return entry, None


def wandb_run_start(run_dir: Path) -> datetime | None:
    m = re.match(r"run-(\d{8})_(\d{6})-", run_dir.name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def log_ts_to_dt(ts: str) -> datetime | None:
    try:
        # Allow suffixes like "_ai2"
        core = ts.split("_ai")[0]
        return datetime.strptime(core, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def main() -> None:
    # Inventory of existing log subdirs
    log_subdirs = []  # list of (log_ts_str, task, mode)
    for ts_dir in LOGS_DIR.iterdir():
        if not ts_dir.is_dir():
            continue
        log_ts_str = ts_dir.name
        log_dt = log_ts_to_dt(log_ts_str)
        if log_dt is None:
            continue
        for sub in ts_dir.iterdir():
            if not sub.is_dir() or not (sub / "server_eval.tsv").exists():
                continue
            # Parse "task_mode" — mode may contain dashes (fedalc-ap-multi)
            name = sub.name
            # Task is the first token before first underscore
            m = re.match(r"^([a-z0-9]+)_(.+)$", name)
            if not m:
                continue
            task = m.group(1)
            mode = m.group(2)
            log_subdirs.append((log_ts_str, log_dt, task, mode, sub))

    # Scan wandb runs
    wandb_rows = []  # (wandb_start_dt, run_name, task, mode, alpha, rounds)
    bad = []
    for run in sorted(WANDB_DIR.glob("run-*")):
        if not run.is_dir():
            continue
        cfg_path = run / "files" / "config.yaml"
        if not cfg_path.exists():
            continue
        entry, err = parse_wandb_config(cfg_path)
        if err:
            bad.append((run.name, err))
            continue
        wst = wandb_run_start(run)
        if wst is None:
            bad.append((run.name, "cannot parse wandb start time"))
            continue
        wandb_rows.append(
            (wst, run.name, str(entry["task"]), str(entry["mode"]),
             float(entry["alpha"]), int(entry["rounds"] or 0))
        )

    # Pair each wandb run to the nearest-earlier log subdir with matching (task, mode)
    # "targets" maps (log_ts, task, mode) -> list of (wandb_start, alpha, run_name, rounds)
    targets = defaultdict(list)
    unmatched_wandb = []
    for wst, run_name, task, mode, alpha, rounds in wandb_rows:
        candidates = [
            (log_ts_str, log_dt, t, m, path)
            for (log_ts_str, log_dt, t, m, path) in log_subdirs
            if t == task and m == mode and log_dt <= wst and (wst - log_dt) <= MATCH_WINDOW
        ]
        if not candidates:
            unmatched_wandb.append((run_name, task, mode, alpha, wst))
            continue
        # Closest = largest log_dt
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = candidates[0]
        targets[(chosen[0], chosen[2], chosen[3])].append((wst, alpha, run_name, rounds))

    # Write inventory TSV (per wandb run)
    OUT_INVENTORY.parent.mkdir(exist_ok=True)
    with OUT_INVENTORY.open("w") as f:
        f.write("log_timestamp\ttask\tmode\talpha\trounds\twandb_run\twandb_start\tstatus\n")
        for key, rows in sorted(targets.items()):
            rows.sort()
            for i, (wst, alpha, run_name, rounds) in enumerate(rows):
                status = "LAST_WRITER" if i == len(rows) - 1 else "overwritten"
                f.write(f"{key[0]}\t{key[1]}\t{key[2]}\t{alpha}\t{rounds}\t"
                        f"{run_name}\t{wst.strftime('%Y%m%d_%H%M%S')}\t{status}\n")
        # Unmatched wandb
        for (run_name, task, mode, alpha, wst) in unmatched_wandb:
            f.write(f"(NO_LOG_MATCH)\t{task}\t{mode}\t{alpha}\t?\t"
                    f"{run_name}\t{wst.strftime('%Y%m%d_%H%M%S')}\tORPHAN\n")

    # Rename manifest: each existing log subdir → new name with alpha tag
    with OUT_MANIFEST.open("w") as f:
        f.write("old_path\tnew_path\talpha\tsource\tnote\n")
        matched_subdir_paths = set()
        for (log_ts, task, mode), rows in sorted(targets.items()):
            rows.sort()
            alpha = rows[-1][1]  # last writer
            alphas = sorted({r[1] for r in rows})
            old = f"logs/{log_ts}/{task}_{mode}"
            new = f"logs/{log_ts}/{task}_{mode}_a{alpha}"
            matched_subdir_paths.add(old)
            note = ""
            if len(alphas) > 1:
                note = f"OVERWRITE alphas seen: {alphas}; surviving data = {alpha}"
            old_exists = (REPO / old).exists()
            source = "wandb"
            if not old_exists:
                note = f"(log dir missing); {note}".strip("; ")
            f.write(f"{old}\t{new}\t{alpha}\t{source}\t{note}\n")

        # Log subdirs with no wandb match — alpha unknown
        for (log_ts, _, task, mode, path) in log_subdirs:
            old_rel = f"logs/{log_ts}/{task}_{mode}"
            if old_rel in matched_subdir_paths:
                continue
            f.write(f"{old_rel}\tSKIP\tUNKNOWN\tnone\tno wandb match (older pre-instrumented run)\n")

    # Console summary
    print(f"Wandb runs scanned: {len(wandb_rows)}  (bad={len(bad)}, orphans={len(unmatched_wandb)})")
    print(f"Log subdirs discovered: {len(log_subdirs)}  (matched: {len(targets)})")
    print(f"Inventory: {OUT_INVENTORY}")
    print(f"Manifest:  {OUT_MANIFEST}")

    # Highlight overwrites
    overwrites = [(k, rows) for k, rows in targets.items() if len({r[1] for r in rows}) > 1]
    if overwrites:
        print(f"\n=== OVERWRITE COLLISIONS: {len(overwrites)} ===")
        for (log_ts, task, mode), rows in overwrites:
            alphas = sorted({r[1] for r in rows})
            last_alpha = sorted(rows)[-1][1]
            print(f"  logs/{log_ts}/{task}_{mode}/  alphas seen={alphas}  surviving={last_alpha}")


if __name__ == "__main__":
    main()
