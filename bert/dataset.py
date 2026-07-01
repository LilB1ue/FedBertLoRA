"""Sequence-classification dataset loading with Dirichlet non-IID partitioning."""

import warnings

from transformers import AutoTokenizer, DataCollatorWithPadding

from bert.experiment_config import build_fds_cache_key
from bert.task_registry import get_legacy_task_config, get_num_labels, get_task_spec

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global cache for FederatedDataset (initialized once per process)
_fds_cache = {}


# Legacy name kept for existing diagnostics and scripts.
GLUE_TASK_CONFIG = get_legacy_task_config()


def _ensure_class_label(dataset, label_col: str, num_labels: int):
    """Cast integer label features to ClassLabel without remapping label IDs."""
    from datasets import ClassLabel

    if isinstance(dataset.features[label_col], ClassLabel):
        return dataset
    return dataset.cast_column(label_col, ClassLabel(num_classes=num_labels))


def _load_partition_with_context(
    fds,
    partition_id: int,
    task_name: str,
    num_partitions: int,
    dirichlet_alpha: float,
    min_partition_size: int,
    partition_failure_hint: str | None = None,
):
    """Load a federated partition and add task config context to retry-limit errors."""
    try:
        return fds.load_partition(partition_id)
    except ValueError as exc:
        if "max number of attempts" not in str(exc):
            raise

        hint = "Lower min-partition-size or increase dirichlet-alpha/num-clients."
        if partition_failure_hint:
            hint = f"{hint} {partition_failure_hint}"
        raise ValueError(
            f"Dirichlet partitioning failed for task-name={task_name!r}, "
            f"partition_id={partition_id}, num_partitions={num_partitions}, "
            f"dirichlet_alpha={dirichlet_alpha}, "
            f"min_partition_size={min_partition_size}. {hint}"
        ) from exc


def load_data(
    partition_id: int,
    num_partitions: int,
    task_name: str,
    model_name: str,
    dirichlet_alpha: float = 0.5,
    min_partition_size: int = 10,
    max_seq_length: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Load a partition of sequence-classification data for federated learning.

    Args:
        partition_id: Client partition index.
        num_partitions: Total number of partitions.
        task_name: Task name (for example sst2, qnli, 20newsgroups).
        model_name: HuggingFace model name for tokenizer.
        dirichlet_alpha: Dirichlet distribution alpha for non-IID split.
        min_partition_size: Minimum number of examples per partition.
        max_seq_length: Maximum token sequence length.
        test_size: Fraction of data for local validation.

    Returns:
        (train_dataset, eval_dataset, tokenizer, data_collator): HF Datasets + tokenizer + collator.
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    task_spec = get_task_spec(task_name)
    cache_key = build_fds_cache_key(
        task_name, num_partitions, dirichlet_alpha, seed, min_partition_size
    )

    global _fds_cache
    if cache_key not in _fds_cache:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=task_spec.label_field,
            alpha=dirichlet_alpha,
            min_partition_size=min_partition_size,
            seed=seed,
        )

        if task_spec.subset:
            _fds_cache[cache_key] = FederatedDataset(
                dataset=task_spec.dataset,
                subset=task_spec.subset,
                partitioners={task_spec.train_split: partitioner},
            )
        else:
            _fds_cache[cache_key] = FederatedDataset(
                dataset=task_spec.dataset,
                partitioners={task_spec.train_split: partitioner},
            )

    fds = _fds_cache[cache_key]
    partition = _load_partition_with_context(
        fds,
        partition_id=partition_id,
        task_name=task_name,
        num_partitions=num_partitions,
        dirichlet_alpha=dirichlet_alpha,
        min_partition_size=min_partition_size,
        partition_failure_hint=task_spec.partition_failure_hint,
    )
    partition = _ensure_class_label(partition, task_spec.label_field, task_spec.num_labels)

    # Stratified train/test split (preserve label distribution)
    # Handle rare labels (count < 2): put them in train only, stratify the rest
    from collections import Counter
    from datasets import concatenate_datasets

    label_col = task_spec.label_field
    label_counts = Counter(partition[label_col])
    rare_labels = {lab for lab, cnt in label_counts.items() if cnt < 2}

    if rare_labels:
        labels = partition[label_col]
        rare_mask = [lab in rare_labels for lab in labels]
        main_indices = [i for i, is_rare in enumerate(rare_mask) if not is_rare]
        rare_indices = [i for i, is_rare in enumerate(rare_mask) if is_rare]

        main_dataset = partition.select(main_indices)
        rare_dataset = partition.select(rare_indices)

        split = main_dataset.train_test_split(
            test_size=test_size, seed=seed, stratify_by_column=label_col,
        )
        split["train"] = concatenate_datasets([split["train"], rare_dataset])
        partition_split = split
    else:
        partition_split = partition.train_test_split(
            test_size=test_size, seed=seed, stratify_by_column=label_col,
        )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_fields = task_spec.text_fields

    def tokenize_fn(examples):
        if len(text_fields) == 1:
            return tokenizer(
                examples[text_fields[0]],
                truncation=True,
                max_length=max_seq_length,
            )
        else:
            return tokenizer(
                examples[text_fields[0]],
                examples[text_fields[1]],
                truncation=True,
                max_length=max_seq_length,
            )

    partition_split = partition_split.map(tokenize_fn, batched=True)

    # Remove text columns, keep only tokenized + label
    columns_to_keep = {"input_ids", "attention_mask", "token_type_ids", label_col, "labels"}
    columns_to_remove = [c for c in partition_split["train"].column_names
                         if c not in columns_to_keep]
    partition_split = partition_split.remove_columns(columns_to_remove)

    # Rename label to labels if needed
    if label_col in partition_split["train"].column_names and label_col != "labels":
        partition_split = partition_split.rename_column(label_col, "labels")

    partition_split.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return partition_split["train"], partition_split["test"], tokenizer, data_collator


def load_centralized_data(
    task_name: str,
    model_name: str,
    max_seq_length: int = 128,
):
    """Load full sequence-classification dataset for centralized training/evaluation.

    Returns:
        (train_dataset, eval_dataset, tokenizer, data_collator): HF Datasets + tokenizer + collator.
    """
    from datasets import load_dataset

    task_spec = get_task_spec(task_name)
    text_fields = task_spec.text_fields

    if task_spec.subset:
        dataset = load_dataset(task_spec.dataset, task_spec.subset)
    else:
        dataset = load_dataset(task_spec.dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        if len(text_fields) == 1:
            return tokenizer(
                examples[text_fields[0]],
                truncation=True,
                max_length=max_seq_length,
            )
        else:
            return tokenizer(
                examples[text_fields[0]],
                examples[text_fields[1]],
                truncation=True,
                max_length=max_seq_length,
            )

    dataset = dataset.map(tokenize_fn, batched=True)

    train_split = task_spec.train_split
    eval_split = task_spec.eval_split
    label_col = task_spec.label_field

    columns_to_keep = {"input_ids", "attention_mask", "token_type_ids", label_col, "labels"}
    columns_to_remove = [c for c in dataset[train_split].column_names
                         if c not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)

    if label_col in dataset[train_split].column_names and label_col != "labels":
        dataset = dataset.rename_column(label_col, "labels")

    dataset.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return dataset[train_split], dataset[eval_split], tokenizer, data_collator
