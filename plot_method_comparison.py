"""Generate method comparison plots using CLIENT-SIDE eval_metrics.tsv.

Rationale: this project targets personalized federated models (each client
owns a customised B + classifier head). Server-side server_eval.tsv applies
Flower's average-of-parameters eval, which is meaningful for global methods
(FedAvg) but misleading for personalized ones (FedSA averages task-specific
B -> noise). Using eval_metrics.tsv (Flower evaluate protocol on each
client's local test split) gives a fair end-user view across methods.

Two aggregations:
  - Unweighted per-client mean  (primary, personalized fairness)
  - Weighted by num_examples    (secondary, == global pooled accuracy)

Outputs per alpha group (α ∈ {0.5, 0.3}) × task (sst2, qnli):
  - all_methods_unweighted_<task>_a<alpha>.png
  - all_methods_weighted_<task>_a<alpha>.png
  - all_methods_best_bar_<task>_a<alpha>.png  (both metrics side-by-side)
  - all_methods_perclient_<task>_a<alpha>.png (boxplot per method at best round)

Usage: python3 plot_method_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGS = Path("logs")
PLOTS_DIR = Path("plots/r30_c30")

# (display_name, batch_dir, subdir, colour, linestyle, note)
METHODS: dict[str, dict[str, list]] = {
    "0.5": {
        "sst2": [
            ("FedAvg",         "20260402_132128_fedavg_a0.5",      "sst2_fedavg_a0.5",      "#1f77b4", "-", None),
            ("FedSA",          "20260405_071932_fedsa_a0.5",       "sst2_fedsa_a0.5",       "#2ca02c", "-", None),
            ("FFA",            "20260420_120302_ffa_a0.5",         "sst2_ffa_a0.5",         "#d62728", "-", None),
            ("FedALC-AP",      "20260406_203614_fedalc_a0.5",      "sst2_fedalc_a0.5",      "#9467bd", "-", None),
            ("FedALC-AP-LWC",  "20260415_063849_fedalc-lwc_a0.5",  "sst2_fedalc-lwc_a0.5",  "#ff7f0e", "--", None),
        ],
        "qnli": [
            ("FedAvg",         "20260402_132128_fedavg_a0.5",      "qnli_fedavg_a0.5",      "#1f77b4", "-", None),
            ("FedSA",          "20260405_071932_fedsa_a0.5",       "qnli_fedsa_a0.5",       "#2ca02c", "-", None),
            ("FFA",            "20260416_064902_ffa_a0.5",         "qnli_ffa_a0.5",         "#d62728", "-", "server-only"),
            ("FedALC-AP",      "20260406_203614_fedalc_a0.5",      "qnli_fedalc_a0.5",      "#9467bd", "-", None),
            ("FedALC-AP-LWC",  "20260415_063849_fedalc-lwc_a0.5",  "qnli_fedalc-lwc_a0.5",  "#ff7f0e", "--", None),
        ],
    },
    "0.3": {
        "sst2": [
            ("FedAvg",         "20260412_235402_fedavg_a0.3",      "sst2_fedavg_a0.3",      "#1f77b4", "-", None),
            ("FedSA",          "20260412_235402_fedsa_a0.3",       "sst2_fedsa_a0.3",       "#2ca02c", "-", None),
            ("FFA",            "20260416_064902_ffa_a0.3",         "sst2_ffa_a0.3",         "#d62728", "-", None),
            ("FedALC-AP",      "20260408_114021_fedalc_a0.3",      "sst2_fedalc_a0.3",      "#9467bd", "-", None),
        ],
        "qnli": [
            ("FedAvg",         "20260412_235402_fedavg_a0.3",      "qnli_fedavg_a0.3",      "#1f77b4", "-", None),
            ("FedSA",          "20260412_235402_fedsa_a0.3",       "qnli_fedsa_a0.3",       "#2ca02c", "-", None),
            ("FFA",            "20260416_064902_ffa_a0.3",         "qnli_ffa_a0.3",         "#d62728", "-", None),
            ("FedALC-AP",      "20260408_114021_fedalc_a0.3",      "qnli_fedalc_a0.3",      "#9467bd", "-", None),
        ],
    },
}


def load_client_eval(batch: str, sub: str) -> pd.DataFrame:
    path = LOGS / batch / sub / "eval_metrics.tsv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    required = {"round", "partition_id", "num_examples", "accuracy"}
    if not required.issubset(df.columns):
        print(f"  WARN: {path} missing columns; found {list(df.columns)}")
        return pd.DataFrame()
    return df


def per_round_stats(df: pd.DataFrame):
    """Returns per-round stats: (round, unweighted_mean, weighted_mean, std, min, max)."""
    records = []
    for rnd, g in df.groupby("round"):
        accs = g["accuracy"].values
        weights = g["num_examples"].values.astype(float)
        unw = accs.mean()
        w = np.average(accs, weights=weights) if weights.sum() > 0 else unw
        records.append((int(rnd), unw, w, accs.std(ddof=0), accs.min(), accs.max()))
    return pd.DataFrame(records, columns=["round", "unweighted", "weighted", "std", "min", "max"])


def plot_metric_curves(alpha: str, task: str, metric: str, out_path: Path) -> list:
    """metric = 'unweighted' or 'weighted'."""
    fig, ax = plt.subplots(figsize=(9, 5))
    summary = []

    for name, batch, sub, colour, style, note in METHODS[alpha][task]:
        df = load_client_eval(batch, sub)
        if df.empty:
            label = f"{name} (no client-side log)"
            ax.plot([], [], label=label, color=colour, linestyle=style, linewidth=1.8)
            summary.append((name, None, None, None, note or "missing"))
            continue
        stats = per_round_stats(df)
        best_idx = stats[metric].idxmax()
        best = stats.loc[best_idx, metric]
        best_round = int(stats.loc[best_idx, "round"])
        last = stats[metric].iloc[-1]
        label = f"{name}{' (' + note + ')' if note else ''}  best={best*100:.2f}%@R{best_round}"
        ax.plot(stats["round"], stats[metric] * 100,
                label=label, color=colour, linestyle=style, linewidth=1.8)
        if metric == "unweighted":
            ax.fill_between(stats["round"],
                            (stats[metric] - stats["std"]) * 100,
                            (stats[metric] + stats["std"]) * 100,
                            color=colour, alpha=0.08)
        # Mark best-round peak
        ax.scatter([best_round], [best * 100], color=colour, s=70,
                   marker="*", edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(f"R{best_round}",
                    xy=(best_round, best * 100),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=colour, fontweight="bold")
        summary.append((name, best * 100, best_round, last * 100, note or ""))

    ax.set_xlabel("Round")
    ax.set_ylabel(f"{'Unweighted' if metric == 'unweighted' else 'Weighted'} per-client accuracy (%)")
    subtitle = "mean ± std" if metric == "unweighted" else "Σ(acc·n) / Σn (== global pooled)"
    ax.set_title(f"{task.upper()} α={alpha} — {metric} per-client acc ({subtitle})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    label_metric = metric.capitalize()
    print(f"\n[α={alpha} {task.upper()}] {label_metric} best per-client accuracy:")
    print(f"  {'Method':<18} {'Best%':>8} {'Round':>6} {'Last%':>8}  Note")
    for name, best, br, last, note in sorted(summary, key=lambda r: -(r[1] if r[1] is not None else -1)):
        if best is None:
            print(f"  {name:<18} {'-':>8} {'-':>6} {'-':>8}  {note}")
            continue
        print(f"  {name:<18} {best:>8.2f} {br:>6d} {last:>8.2f}  {note}")
    return summary


def plot_best_bar_combined(alpha: str, task: str, unw_rows, w_rows, out_path: Path) -> None:
    """Side-by-side bars: unweighted vs weighted best."""
    names = [r[0] for r in unw_rows]
    unw_map = {r[0]: r[1] for r in unw_rows}
    w_map = {r[0]: r[1] for r in w_rows}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.36
    u_vals = [unw_map.get(n) or 0 for n in names]
    w_vals = [w_map.get(n) or 0 for n in names]

    b1 = ax.bar(x - width / 2, u_vals, width, label="Unweighted mean", color="#1f77b4", alpha=0.85)
    b2 = ax.bar(x + width / 2, w_vals, width, label="Weighted mean", color="#ff7f0e", alpha=0.85)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("Best per-client accuracy (%)")
    ax.set_title(f"Best per-client acc — {task.upper()} α={alpha}  (unweighted vs weighted)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    ax.set_ylim(min([v for v in u_vals + w_vals if v > 0] + [0]) - 5, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_perclient_distribution(alpha: str, task: str, out_path: Path) -> None:
    """Boxplot of per-client accuracy at each method's best round."""
    data, labels, colours = [], [], []
    for name, batch, sub, colour, _, note in METHODS[alpha][task]:
        df = load_client_eval(batch, sub)
        if df.empty:
            continue
        stats = per_round_stats(df)
        best_round = int(stats.loc[stats["unweighted"].idxmax(), "round"])
        accs = df[df["round"] == best_round]["accuracy"].values * 100
        data.append(accs)
        labels.append(name + (f"\n({note})" if note else ""))
        colours.append(colour)

    if not data:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    medianprops={"color": "black"})
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_ylabel("Per-client accuracy at best round (%)")
    ax.set_title(f"Per-client accuracy distribution — {task.upper()} α={alpha}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for alpha in ("0.5", "0.3"):
        for task in ("sst2", "qnli"):
            unw = plot_metric_curves(alpha, task, "unweighted",
                                     PLOTS_DIR / f"all_methods_unweighted_{task}_a{alpha}.png")
            w = plot_metric_curves(alpha, task, "weighted",
                                   PLOTS_DIR / f"all_methods_weighted_{task}_a{alpha}.png")
            # Bar plot skipped per project preference (plots/README.md: line + box only).
            # plot_best_bar_combined stays defined in case needed later.
            _ = (unw, w)
            plot_perclient_distribution(alpha, task,
                                        PLOTS_DIR / f"all_methods_perclient_{task}_a{alpha}.png")
    print("\nDone. Plots in plots/r30_c30/all_methods_{unweighted,weighted,perclient}_*.png (bar plot type archived)")


if __name__ == "__main__":
    main()
