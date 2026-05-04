"""Dump per-partition train/test sizes and label distributions.

Output per (task, num_partitions, alpha, seed) combination:
  - JSON: `logs/partition_stats/{task}_c{N}_a{alpha}_s{seed}.json`
    schema:
      {
        "meta": {task, num_partitions, alpha, seed, test_split_ratio, total_train, total_test},
        "partitions": [
          {
            "partition_id": 0,
            "n_train": 707, "n_test": 177,
            "label_counts_train": {"0": 301, "1": 406},
            "label_counts_test": {"0": 75, "1": 102},
            "label_ratio_train": {"0": 0.4258, "1": 0.5742},
          },
          ...
        ],
      }
  - TSV (flattened): same stem, `.tsv` extension
      partition_id  n_train  n_test  label_0_train  label_1_train  ...  label_0_ratio  label_1_ratio

Uses the same DirichletPartitioner + seeded train/test split as bert/dataset.py
so stats match exactly what training actually sees. Does NOT tokenize (faster).

Usage:
  python tools/dump_partition_stats.py --task sst2 --alpha 0.5
  python tools/dump_partition_stats.py --task qnli --alpha 0.3 --num-partitions 30
  # Batch all four combos at once:
  for t in sst2 qnli; do for a in 0.3 0.5; do
    python tools/dump_partition_stats.py --task $t --alpha $a
  done; done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Make the repo root importable (for `from bert.dataset import GLUE_TASK_CONFIG`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bert.dataset import GLUE_TASK_CONFIG


def compute_partition_stats(
    task: str,
    num_partitions: int,
    alpha: float,
    seed: int,
    test_split_ratio: float,
):
    """Replicate bert.dataset.load_data's partitioning + stratified split; no tokenization."""
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner
    from datasets import concatenate_datasets

    task_cfg = GLUE_TASK_CONFIG[task]
    label_col = task_cfg["label_field"]

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by=label_col,
        alpha=alpha,
        min_partition_size=10,
        seed=seed,
    )

    dataset_name = task_cfg["dataset"]
    subset = task_cfg.get("subset")
    if subset:
        fds = FederatedDataset(
            dataset=dataset_name, subset=subset,
            partitioners={"train": partitioner},
        )
    else:
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    entries = []
    total_train = 0
    total_test = 0

    for pid in range(num_partitions):
        partition = fds.load_partition(pid)

        # Replicate stratified split from dataset.py
        label_counts = Counter(partition[label_col])
        rare_labels = {lab for lab, cnt in label_counts.items() if cnt < 2}

        if rare_labels:
            labels_raw = partition[label_col]
            rare_mask = [lab in rare_labels for lab in labels_raw]
            main_indices = [i for i, is_rare in enumerate(rare_mask) if not is_rare]
            rare_indices = [i for i, is_rare in enumerate(rare_mask) if is_rare]
            main = partition.select(main_indices)
            rare = partition.select(rare_indices)
            split = main.train_test_split(
                test_size=test_split_ratio, seed=seed,
                stratify_by_column=label_col,
            )
            split["train"] = concatenate_datasets([split["train"], rare])
        else:
            split = partition.train_test_split(
                test_size=test_split_ratio, seed=seed,
                stratify_by_column=label_col,
            )

        train_labels = split["train"][label_col]
        test_labels = split["test"][label_col]
        lc_train = Counter(train_labels)
        lc_test = Counter(test_labels)
        n_train = len(train_labels)
        n_test = len(test_labels)

        entry = {
            "partition_id": pid,
            "n_train": n_train,
            "n_test": n_test,
            "label_counts_train": {str(k): int(v) for k, v in sorted(lc_train.items())},
            "label_counts_test": {str(k): int(v) for k, v in sorted(lc_test.items())},
            "label_ratio_train": {
                str(k): round(v / n_train, 4) for k, v in sorted(lc_train.items())
            } if n_train else {},
        }
        entries.append(entry)
        total_train += n_train
        total_test += n_test
        print(f"  pid={pid:2d}  n_train={n_train:5d}  n_test={n_test:4d}  "
              f"labels_train={dict(sorted(lc_train.items()))}")

    meta = {
        "task": task,
        "num_partitions": num_partitions,
        "alpha": alpha,
        "seed": seed,
        "test_split_ratio": test_split_ratio,
        "total_train": total_train,
        "total_test": total_test,
    }
    return {"meta": meta, "partitions": entries}


def write_outputs(stats: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = stats["meta"]
    stem = f"{meta['task']}_c{meta['num_partitions']}_a{meta['alpha']}_s{meta['seed']}"

    # JSON
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    # TSV (flattened — assumes consistent label set across partitions)
    all_labels = sorted({
        k
        for e in stats["partitions"]
        for k in set(e["label_counts_train"]) | set(e["label_counts_test"])
    })

    tsv_path = output_dir / f"{stem}.tsv"
    with open(tsv_path, "w") as f:
        header = ["partition_id", "n_train", "n_test"]
        header += [f"label_{L}_train" for L in all_labels]
        header += [f"label_{L}_test" for L in all_labels]
        header += [f"label_{L}_ratio_train" for L in all_labels]
        f.write("\t".join(header) + "\n")

        for e in stats["partitions"]:
            row = [str(e["partition_id"]), str(e["n_train"]), str(e["n_test"])]
            row += [str(e["label_counts_train"].get(L, 0)) for L in all_labels]
            row += [str(e["label_counts_test"].get(L, 0)) for L in all_labels]
            row += [str(e["label_ratio_train"].get(L, 0.0)) for L in all_labels]
            f.write("\t".join(row) + "\n")

    return json_path, tsv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(GLUE_TASK_CONFIG.keys()))
    parser.add_argument("--num-partitions", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="logs/partition_stats")
    args = parser.parse_args()

    print(f"Computing partition stats: task={args.task} N={args.num_partitions} "
          f"α={args.alpha} seed={args.seed}")
    stats = compute_partition_stats(
        task=args.task,
        num_partitions=args.num_partitions,
        alpha=args.alpha,
        seed=args.seed,
        test_split_ratio=args.test_split_ratio,
    )

    json_path, tsv_path = write_outputs(stats, Path(args.output_dir))

    meta = stats["meta"]
    print(f"\n=== Summary ===")
    print(f"  total_train = {meta['total_train']}")
    print(f"  total_test  = {meta['total_test']}")
    print(f"  n_train per partition: min={min(e['n_train'] for e in stats['partitions'])}, "
          f"max={max(e['n_train'] for e in stats['partitions'])}")
    print(f"\nOutputs:\n  {json_path}\n  {tsv_path}")


if __name__ == "__main__":
    main()
