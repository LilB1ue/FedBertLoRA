"""Small helpers for experiment config and checkpoint retention policy."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


SUPPORTED_BEST_METRICS = {
    "sst2": {"accuracy"},
    "qnli": {"accuracy"},
    "mnli": {"accuracy"},
    "rte": {"accuracy"},
    "qqp": {"accuracy", "f1"},
}


def build_fds_cache_key(
    task_name: str,
    num_partitions: int,
    dirichlet_alpha: float,
    seed: int,
    min_partition_size: int,
) -> str:
    """Return the FederatedDataset cache key for a partition config."""
    return f"{task_name}_{num_partitions}_{dirichlet_alpha}_{seed}_min{min_partition_size}"


def build_alpha_tag(alpha: float) -> str:
    """Return the canonical alpha suffix used in log directories."""
    return f"_a{float(alpha)}"


def build_batch_dir(timestamp: str, aggregation_mode: str, alpha: float) -> str:
    """Return the canonical outer batch directory name."""
    return f"{timestamp}_{aggregation_mode}{build_alpha_tag(alpha)}"


def build_run_subdir(task_name: str, aggregation_mode: str, alpha: float) -> str:
    """Return the canonical inner run directory name."""
    return f"{task_name}_{aggregation_mode}{build_alpha_tag(alpha)}"


def build_run_dir(
    log_dir: str,
    timestamp: str,
    task_name: str,
    aggregation_mode: str,
    alpha: float,
) -> str:
    """Return the canonical full run directory path."""
    return os.path.join(
        log_dir,
        build_batch_dir(timestamp, aggregation_mode, alpha),
        build_run_subdir(task_name, aggregation_mode, alpha),
    )


def build_wandb_run_name(
    task_name: str,
    aggregation_mode: str,
    num_clients: int,
    num_rounds: int,
    alpha: float,
    host: str,
    timestamp: str,
    run_tag: str = "",
) -> str:
    """Return a W&B run name with an optional experiment protocol tag."""
    prefix_parts = [task_name, aggregation_mode]
    cleaned_tag = str(run_tag).strip()
    if cleaned_tag:
        prefix_parts.append(cleaned_tag)
    prefix = "_".join(prefix_parts)
    return f"{prefix}_c{num_clients}_r{num_rounds}_a{alpha}_{host}_{timestamp}"


def validate_checkpoint_best_metric(task_name: str, metric: str) -> None:
    """Fail fast if a task will not emit the configured best-checkpoint metric."""
    supported = SUPPORTED_BEST_METRICS.get(task_name)
    if supported is None:
        raise ValueError(f"Unknown task-name {task_name!r}; cannot validate checkpoint-best-metric")
    if metric not in supported:
        supported_list = ", ".join(sorted(supported))
        raise ValueError(
            f"checkpoint-best-metric={metric!r} is not supported for task-name={task_name!r}; "
            f"supported metrics: {supported_list}"
        )


def normalize_checkpoint_policy(policy: str) -> str:
    """Validate and normalize a checkpoint retention policy."""
    normalized = str(policy).strip().lower()
    if normalized not in {"all", "selective"}:
        raise ValueError(
            "checkpoint-save-policy must be 'all' or 'selective', "
            f"got {policy!r}"
        )
    return normalized


def should_save_client_checkpoint(policy: str, current_round: int) -> bool:
    """Return whether to save post-training client checkpoint for this round."""
    policy = normalize_checkpoint_policy(policy)
    return policy == "all" or current_round == 1


def should_save_received_checkpoint(
    policy: str,
    aggregation_mode: str,
    current_round: int,
) -> bool:
    """Return whether to save pre-training received checkpoint for this round."""
    if aggregation_mode == "fedavg":
        return False
    return should_save_client_checkpoint(policy, current_round)


def should_save_global_checkpoint(policy: str, aggregation_mode: str) -> bool:
    """Return whether to save legacy FedAvg server-side global checkpoints."""
    policy = normalize_checkpoint_policy(policy)
    return policy == "all" and aggregation_mode == "fedavg"


class BestCheckpointTracker:
    """Track and prune evaluation-aligned best checkpoint candidates.

    Candidate checkpoints are expected at ``root_dir/round_R/client_ID`` before
    ``update`` is called. ``update`` keeps only the best round directory and a
    ``best_round.json`` metadata file.
    """

    def __init__(self, root_dir: str | Path, metric: str, mode: str = "max") -> None:
        self.root_dir = Path(root_dir)
        self.metric = metric
        self.mode = mode.lower()
        if self.mode not in {"max", "min"}:
            raise ValueError(f"checkpoint-best-mode must be 'max' or 'min', got {mode!r}")
        self.best_metadata: Dict[str, float | int | str] | None = None

    def update(
        self,
        server_round: int,
        metrics_list: List[Tuple[int, dict]],
    ) -> Dict[str, float | int | str]:
        """Update best round from client-side evaluation metrics."""
        metadata = self._metadata_for_round(server_round, metrics_list)

        if self.best_metadata is None or self._is_better(metadata["value"], self.best_metadata["value"]):
            previous_round = None if self.best_metadata is None else int(self.best_metadata["round"])
            self.best_metadata = metadata
            self._write_metadata(metadata)
            if previous_round is not None and previous_round != server_round:
                self._remove_round(previous_round)
        else:
            self._remove_round(server_round)

        return dict(self.best_metadata)

    def _metadata_for_round(
        self,
        server_round: int,
        metrics_list: List[Tuple[int, dict]],
    ) -> Dict[str, float | int | str]:
        values = [
            (num_examples, metrics[self.metric])
            for num_examples, metrics in metrics_list
            if isinstance(metrics.get(self.metric), (int, float))
        ]
        if not values:
            raise ValueError(f"No numeric metric {self.metric!r} found for round {server_round}")

        unweighted = sum(value for _, value in values) / len(values)
        total_examples = sum(num_examples for num_examples, _ in values)
        weighted = (
            sum(num_examples * value for num_examples, value in values) / total_examples
            if total_examples > 0
            else unweighted
        )

        return {
            "round": int(server_round),
            "metric": self.metric,
            "mode": self.mode,
            "selection": "unweighted_mean",
            "value": float(unweighted),
            "weighted_mean": float(weighted),
            "num_clients": len(values),
        }

    def _is_better(self, value: float, best_value: float) -> bool:
        if self.mode == "max":
            return value > best_value
        return value < best_value

    def _write_metadata(self, metadata: Dict[str, float | int | str]) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with open(self.root_dir / "best_round.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

    def _remove_round(self, server_round: int) -> None:
        round_dir = self.root_dir / f"round_{server_round}"
        if round_dir.exists():
            shutil.rmtree(round_dir)
