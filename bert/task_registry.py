"""Task registry for sequence-classification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    text_fields: Tuple[str, ...]
    label_field: str
    num_labels: int
    subset: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    metric_name: str = "accuracy"
    metric_config: Optional[str] = None
    partition_failure_hint: Optional[str] = None

    def as_legacy_dict(self) -> dict:
        """Return the old dict shape used by diagnostics and older scripts."""
        data = {
            "dataset": self.dataset,
            "text_fields": list(self.text_fields),
            "label_field": self.label_field,
            "num_labels": self.num_labels,
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "metric_name": self.metric_name,
            "metric_config": self.metric_config,
            "partition_failure_hint": self.partition_failure_hint,
        }
        if self.subset is not None:
            data["subset"] = self.subset
        return data


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "sst2": TaskSpec(
        dataset="stanfordnlp/sst2",
        text_fields=("sentence",),
        label_field="label",
        num_labels=2,
        metric_name="glue",
        metric_config="sst2",
    ),
    "qnli": TaskSpec(
        dataset="nyu-mll/glue",
        subset="qnli",
        text_fields=("question", "sentence"),
        label_field="label",
        num_labels=2,
        metric_name="glue",
        metric_config="qnli",
    ),
    "mnli": TaskSpec(
        dataset="nyu-mll/glue",
        subset="mnli",
        text_fields=("premise", "hypothesis"),
        label_field="label",
        num_labels=3,
        eval_split="validation_matched",
        metric_name="glue",
        metric_config="mnli",
    ),
    "qqp": TaskSpec(
        dataset="nyu-mll/glue",
        subset="qqp",
        text_fields=("question1", "question2"),
        label_field="label",
        num_labels=2,
        metric_name="glue",
        metric_config="qqp",
    ),
    "rte": TaskSpec(
        dataset="nyu-mll/glue",
        subset="rte",
        text_fields=("sentence1", "sentence2"),
        label_field="label",
        num_labels=2,
        metric_name="glue",
        metric_config="rte",
    ),
    "20newsgroups": TaskSpec(
        dataset="SetFit/20_newsgroups",
        text_fields=("text",),
        label_field="label",
        num_labels=20,
        train_split="train",
        eval_split="test",
        metric_name="accuracy",
        partition_failure_hint=(
            "For 20newsgroups with 30 clients at alpha=0.3, "
            "min-partition-size=128 was verified; 256 failed."
        ),
    ),
}


def get_task_names() -> tuple[str, ...]:
    """Return supported task names in config order."""
    return tuple(TASK_REGISTRY.keys())


def get_task_spec(task_name: str) -> TaskSpec:
    """Return the task spec for a supported sequence-classification task."""
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        supported = ", ".join(get_task_names())
        raise KeyError(f"Unknown task-name {task_name!r}; supported tasks: {supported}") from exc


def get_num_labels(task_name: str) -> int:
    """Return number of labels for a supported task."""
    return get_task_spec(task_name).num_labels


def get_legacy_task_config() -> dict[str, dict]:
    """Return task config in the old dict-of-dicts shape."""
    return {name: spec.as_legacy_dict() for name, spec in TASK_REGISTRY.items()}


def _load_metric(metric_name: str, metric_config: str | None = None):
    from evaluate import load as load_metric

    if metric_config is None:
        return load_metric(metric_name)
    return load_metric(metric_name, metric_config)


def load_metric_for_task(task_name: str):
    """Load the metric configured for a supported task."""
    spec = get_task_spec(task_name)
    if spec.metric_config is None:
        return _load_metric(spec.metric_name)
    return _load_metric(spec.metric_name, spec.metric_config)
