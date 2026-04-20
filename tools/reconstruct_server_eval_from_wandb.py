"""Reconstruct server_eval.tsv for runs whose local logs were overwritten.

Wandb output.log contains lines like:
  [Server] Round 12: loss=0.2031, accuracy=0.9478

Given (wandb_run_dir, target_log_dir), parse those lines and emit server_eval.tsv
in the same format as server_app.py writes (round\taccuracy\tloss).

Usage:
  python3 tools/reconstruct_server_eval_from_wandb.py

Hardcoded targets for now (FFA alpha=0.5 recovery after overwrite).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (wandb_run_name, target_subdir relative to REPO)
TARGETS = [
    ("run-20260416_064907-3sng404d", "logs/20260416_064902/sst2_ffa_a0.5"),
    ("run-20260416_082547-bhv4wt60", "logs/20260416_064902/qnli_ffa_a0.5"),
]

SERVER_RE = re.compile(r"\[Server\] Round (\d+): loss=([0-9.eE+-]+), accuracy=([0-9.eE+-]+)")


def reconstruct(wandb_run: str, target_rel: str) -> None:
    out_log = REPO / "wandb" / wandb_run / "files" / "output.log"
    target_dir = REPO / target_rel
    if not out_log.exists():
        print(f"SKIP: output.log missing for {wandb_run}")
        return
    rows = []
    for line in out_log.read_text(errors="replace").splitlines():
        m = SERVER_RE.search(line)
        if m:
            rnd = int(m.group(1))
            loss = float(m.group(2))
            acc = float(m.group(3))
            rows.append((rnd, acc, loss))
    if not rows:
        print(f"SKIP: no [Server] lines found in {wandb_run}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / "server_eval.tsv"
    with dst.open("w") as f:
        f.write("round\taccuracy\tloss\n")
        for rnd, acc, loss in rows:
            f.write(f"{rnd}\t{acc}\t{loss}\n")

    # Add provenance note
    note = target_dir / "RECONSTRUCTED_FROM_WANDB.txt"
    note.write_text(
        f"This server_eval.tsv was reconstructed from wandb/{wandb_run}/files/output.log\n"
        f"because the original local log was overwritten by a later run with a different alpha.\n"
        f"Only server_eval.tsv is recoverable; fit_metrics/eval_metrics/checkpoints are lost.\n"
        f"Rounds recovered: {len(rows)} (first={rows[0][0]}, last={rows[-1][0]})\n"
        f"Note: accuracy values have limited precision (4 decimals) due to log formatting.\n"
    )
    print(f"OK: {target_rel}/server_eval.tsv  ({len(rows)} rows)")


def main() -> None:
    for wr, target in TARGETS:
        reconstruct(wr, target)


if __name__ == "__main__":
    main()
