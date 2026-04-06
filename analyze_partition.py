"""Analyze data distribution across all federated partitions and output markdown."""

import sys
from collections import Counter

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from bert.dataset import GLUE_TASK_CONFIG


def analyze(task_name="sst2", num_partitions=40, alpha=0.5, test_size=0.2, seed=42):
    task_cfg = GLUE_TASK_CONFIG[task_name]
    label_field = task_cfg["label_field"]
    num_labels = task_cfg["num_labels"]

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by=label_field,
        alpha=alpha,
        min_partition_size=0,
        seed=seed,
    )

    dataset_name = task_cfg["dataset"]
    subset = task_cfg.get("subset")
    if subset:
        fds = FederatedDataset(dataset=dataset_name, subset=subset, partitioners={"train": partitioner})
    else:
        fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})

    # Label names
    if task_name == "sst2":
        label_names = ["negative", "positive"]
    elif task_name in ("qnli", "rte"):
        label_names = ["entailment", "not_entailment"]
    elif task_name == "mnli":
        label_names = ["entailment", "neutral", "contradiction"]
    elif task_name == "qqp":
        label_names = ["not_duplicate", "duplicate"]
    else:
        label_names = [f"label_{i}" for i in range(num_labels)]

    rows = []
    for pid in range(num_partitions):
        partition = fds.load_partition(pid)
        labels = partition[label_field]
        total = len(labels)
        counts = Counter(labels)

        # Compute train/eval split sizes (same logic as dataset.py)
        rare_labels = {lab for lab, cnt in counts.items() if cnt < 2}
        rare_count = sum(counts[lab] for lab in rare_labels)
        main_count = total - rare_count

        eval_count = int(main_count * test_size)
        train_count = total - eval_count  # rare samples go to train

        # Label distribution
        dist = {i: counts.get(i, 0) for i in range(num_labels)}
        dist_pct = {i: counts.get(i, 0) / total * 100 if total > 0 else 0 for i in range(num_labels)}

        rows.append({
            "pid": pid,
            "total": total,
            "train": train_count,
            "eval": eval_count,
            "dist": dist,
            "dist_pct": dist_pct,
        })

    # Sort by total descending
    rows.sort(key=lambda x: x["total"], reverse=True)

    # Generate markdown
    lines = []
    lines.append(f"# Data Distribution: {task_name.upper()}")
    lines.append("")
    lines.append(f"- **Task**: {task_name.upper()} ({task_cfg['dataset']})")
    lines.append(f"- **Num clients**: {num_partitions}")
    lines.append(f"- **Dirichlet alpha**: {alpha}")
    lines.append(f"- **Train/Eval split**: {1-test_size:.0%} / {test_size:.0%}")
    lines.append(f"- **Seed**: {seed}")
    total_all = sum(r["total"] for r in rows)
    lines.append(f"- **Total samples**: {total_all:,}")
    lines.append("")

    # Summary stats
    totals = [r["total"] for r in rows]
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Mean samples/client | {sum(totals)/len(totals):.1f} |")
    lines.append(f"| Min samples | {min(totals)} (client {min(rows, key=lambda x: x['total'])['pid']}) |")
    lines.append(f"| Max samples | {max(totals)} (client {max(rows, key=lambda x: x['total'])['pid']}) |")
    lines.append(f"| Std | {(sum((t - sum(totals)/len(totals))**2 for t in totals) / len(totals))**0.5:.1f} |")
    lines.append("")

    # Per-client table
    lines.append("## Per-Client Data Distribution")
    lines.append("")

    # Header
    dist_headers = " | ".join([f"{name}" for name in label_names])
    pct_headers = " | ".join([f"% {name}" for name in label_names])
    lines.append(f"| Client | Total | Train | Eval | {dist_headers} | {pct_headers} |")
    sep = "|--------|------:|------:|-----:|" + "|".join(["------:" for _ in label_names]) + "|" + "|".join(["------:" for _ in label_names]) + "|"
    lines.append(sep)

    for r in rows:
        dist_vals = " | ".join([f"{r['dist'][i]}" for i in range(num_labels)])
        pct_vals = " | ".join([f"{r['dist_pct'][i]:.1f}" for i in range(num_labels)])
        lines.append(f"| {r['pid']} | {r['total']} | {r['train']} | {r['eval']} | {dist_vals} | {pct_vals} |")

    lines.append("")

    # Overall distribution
    lines.append("## Overall Label Distribution")
    lines.append("")
    overall = Counter()
    for r in rows:
        for i in range(num_labels):
            overall[i] += r["dist"][i]
    lines.append("| Label | Count | % |")
    lines.append("|-------|------:|---:|")
    for i in range(num_labels):
        lines.append(f"| {label_names[i]} | {overall[i]:,} | {overall[i]/total_all*100:.1f} |")
    lines.append("")

    md = "\n".join(lines)
    out_path = f"data_distribution_{task_name}.md"
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Written to {out_path}")
    print(md)


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "sst2"
    analyze(task_name=task)
