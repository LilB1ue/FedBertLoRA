"""Generate method comparison plots across FedAvg / FedSA / FFA / FedALC-AP / FedALC-AP-LWC.

Reads server_eval.tsv from each run (new log subdir convention with _a<alpha> suffix).
Produces four plots per alpha group:
  - plots/r30_c30/all_methods_accuracy_sst2_a<alpha>.png
  - plots/r30_c30/all_methods_accuracy_qnli_a<alpha>.png
  - plots/r30_c30/all_methods_loss_qnli_a<alpha>.png
  - plots/r30_c30/all_methods_best_bar_a<alpha>.png

FFA α=0.5 data is reconstructed from wandb output.log (see
tools/reconstruct_server_eval_from_wandb.py), with 4-decimal precision.

Usage: python3 plot_method_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOGS = Path("logs")
PLOTS_DIR = Path("plots/r30_c30")

# (display_name, run_path, colour, linestyle, note)
# note="reconstructed" means data reconstructed from wandb output.log with 4-dp precision
METHODS: dict[str, dict[str, list]] = {
    "0.5": {
        "sst2": [
            ("FedAvg",         LOGS / "20260402_132128" / "sst2_fedavg_a0.5",      "#1f77b4", "-", None),
            ("FedSA",          LOGS / "20260405_071932" / "sst2_fedsa_a0.5",       "#2ca02c", "-", None),
            ("FFA",            LOGS / "20260416_064902" / "sst2_ffa_a0.5",         "#d62728", "-", "reconstructed"),
            ("FedALC-AP",      LOGS / "20260406_203614" / "sst2_fedalc_a0.5",      "#9467bd", "-", None),
            ("FedALC-AP-LWC",  LOGS / "20260415_063849" / "sst2_fedalc-lwc_a0.5",  "#ff7f0e", "--", None),
        ],
        "qnli": [
            ("FedAvg",         LOGS / "20260402_132128" / "qnli_fedavg_a0.5",      "#1f77b4", "-", None),
            ("FedSA",          LOGS / "20260405_071932" / "qnli_fedsa_a0.5",       "#2ca02c", "-", None),
            ("FFA",            LOGS / "20260416_064902" / "qnli_ffa_a0.5",         "#d62728", "-", "reconstructed"),
            ("FedALC-AP",      LOGS / "20260406_203614" / "qnli_fedalc_a0.5",      "#9467bd", "-", None),
            ("FedALC-AP-LWC",  LOGS / "20260415_063849" / "qnli_fedalc-lwc_a0.5",  "#ff7f0e", "--", None),
        ],
    },
    "0.3": {
        "sst2": [
            ("FedAvg",         LOGS / "20260412_235402" / "sst2_fedavg_a0.3",      "#1f77b4", "-", None),
            ("FedSA",          LOGS / "20260412_235402" / "sst2_fedsa_a0.3",       "#2ca02c", "-", None),
            ("FFA",            LOGS / "20260416_064902" / "sst2_ffa_a0.3",         "#d62728", "-", None),
            ("FedALC-AP",      LOGS / "20260408_114021" / "sst2_fedalc_a0.3",      "#9467bd", "-", None),
        ],
        "qnli": [
            ("FedAvg",         LOGS / "20260412_235402" / "qnli_fedavg_a0.3",      "#1f77b4", "-", None),
            ("FedSA",          LOGS / "20260412_235402" / "qnli_fedsa_a0.3",       "#2ca02c", "-", None),
            ("FFA",            LOGS / "20260416_064902" / "qnli_ffa_a0.3",         "#d62728", "-", None),
            ("FedALC-AP",      LOGS / "20260408_114021" / "qnli_fedalc_a0.3",      "#9467bd", "-", None),
        ],
    },
}


def load_run(run_path: Path) -> pd.DataFrame:
    tsv = run_path / "server_eval.tsv"
    if not tsv.exists():
        print(f"  MISSING: {tsv}")
        return pd.DataFrame()
    return pd.read_csv(tsv, sep="\t")


def plot_accuracy_curves(alpha: str, task: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    summary_rows = []

    for name, run_path, colour, style, note in METHODS[alpha][task]:
        df = load_run(run_path)
        if df.empty:
            continue
        label = name + (f" (*recon)" if note == "reconstructed" else "")
        ax.plot(df["round"], df["accuracy"] * 100,
                label=label, color=colour, linestyle=style, linewidth=1.8)
        best = df["accuracy"].max()
        best_round = int(df.loc[df["accuracy"].idxmax(), "round"])
        last = df["accuracy"].iloc[-1]
        summary_rows.append((name, best * 100, best_round, last * 100, len(df), note))

    ax.set_xlabel("Round")
    ax.set_ylabel("Server-side accuracy (%)")
    ax.set_title(f"Method comparison — {task.upper()} (α={alpha}, 30 clients)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[α={alpha} {task.upper()}] Best server-side accuracy:")
    print(f"  {'Method':<18} {'Best%':>8} {'Round':>6} {'Last%':>8} {'Rounds':>7}  Note")
    for name, best, br, last, nrounds, note in sorted(summary_rows, key=lambda x: -x[1]):
        note_str = note or ""
        print(f"  {name:<18} {best:>8.2f} {br:>6d} {last:>8.2f} {nrounds:>7d}  {note_str}")

    return summary_rows


def plot_qnli_loss(alpha: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, run_path, colour, style, note in METHODS[alpha]["qnli"]:
        df = load_run(run_path)
        if df.empty:
            continue
        label = name + (f" (*recon)" if note == "reconstructed" else "")
        ax.plot(df["round"], df["loss"],
                label=label, color=colour, linestyle=style, linewidth=1.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Server-side eval loss")
    ax.set_title(f"QNLI — server-side loss (α={alpha}, FedSA diverges)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_bar(alpha: str, sst2_rows, qnli_rows, out_path: Path):
    sst2_map = {r[0]: r[1] for r in sst2_rows}
    qnli_map = {r[0]: r[1] for r in qnli_rows}
    methods = [r[0] for r in sst2_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(methods))
    width = 0.36
    sst2_vals = [sst2_map.get(m, 0) for m in methods]
    qnli_vals = [qnli_map.get(m, 0) for m in methods]
    bars1 = ax.bar([i - width / 2 for i in x], sst2_vals,
                   width, label="SST-2", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar([i + width / 2 for i in x], qnli_vals,
                   width, label="QNLI", color="#ff7f0e", alpha=0.85)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=10)
    ax.set_ylabel("Best accuracy (%)")
    ax.set_title(f"Best server-side accuracy (α={alpha}, 30 clients)")
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for alpha in ("0.5", "0.3"):
        sst2_rows = plot_accuracy_curves(alpha, "sst2",
                                         PLOTS_DIR / f"all_methods_accuracy_sst2_a{alpha}.png")
        qnli_rows = plot_accuracy_curves(alpha, "qnli",
                                         PLOTS_DIR / f"all_methods_accuracy_qnli_a{alpha}.png")
        plot_qnli_loss(alpha, PLOTS_DIR / f"all_methods_loss_qnli_a{alpha}.png")
        plot_best_bar(alpha, sst2_rows, qnli_rows,
                      PLOTS_DIR / f"all_methods_best_bar_a{alpha}.png")

    print("\nSaved plots under plots/r30_c30/ with _a<alpha> suffix.")


if __name__ == "__main__":
    main()
