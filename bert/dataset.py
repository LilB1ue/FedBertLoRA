"""GLUE dataset loading with Dirichlet non-IID partitioning for federated learning."""

import warnings

from transformers import AutoTokenizer, DataCollatorWithPadding

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global cache for FederatedDataset (initialized once per process)
_fds_cache = {}


# GLUE task configurations
GLUE_TASK_CONFIG = {
    "sst2": {
        "dataset": "stanfordnlp/sst2",
        "text_fields": ["sentence"],
        "label_field": "label",
        "num_labels": 2,
    },
    "qnli": {
        "dataset": "nyu-mll/glue",
        "subset": "qnli",
        "text_fields": ["question", "sentence"],
        "label_field": "label",
        "num_labels": 2,
    },
    "mnli": {
        "dataset": "nyu-mll/glue",
        "subset": "mnli",
        "text_fields": ["premise", "hypothesis"],
        "label_field": "label",
        "num_labels": 3,
    },
    "qqp": {
        "dataset": "nyu-mll/glue",
        "subset": "qqp",
        "text_fields": ["question1", "question2"],
        "label_field": "label",
        "num_labels": 2,
    },
    "rte": {
        "dataset": "nyu-mll/glue",
        "subset": "rte",
        "text_fields": ["sentence1", "sentence2"],
        "label_field": "label",
        "num_labels": 2,
    },
}


def get_num_labels(task_name: str) -> int:
    """Return number of labels for a GLUE task."""
    return GLUE_TASK_CONFIG[task_name]["num_labels"]


def load_data(
    partition_id: int,
    num_partitions: int,
    task_name: str,
    model_name: str,
    dirichlet_alpha: float = 0.5,
    max_seq_length: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Load a partition of GLUE data for federated learning.

    Args:
        partition_id: Client partition index.
        num_partitions: Total number of partitions.
        task_name: GLUE task name (sst2, qnli, mnli, qqp, rte).
        model_name: HuggingFace model name for tokenizer.
        dirichlet_alpha: Dirichlet distribution alpha for non-IID split.
        max_seq_length: Maximum token sequence length.
        test_size: Fraction of data for local validation.

    Returns:
        (train_dataset, eval_dataset, tokenizer, data_collator): HF Datasets + tokenizer + collator.
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    task_cfg = GLUE_TASK_CONFIG[task_name]
    cache_key = f"{task_name}_{num_partitions}_{dirichlet_alpha}_{seed}"

    global _fds_cache
    if cache_key not in _fds_cache:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=task_cfg["label_field"],
            alpha=dirichlet_alpha,
            min_partition_size=0,
            seed=seed,
        )

        dataset_name = task_cfg["dataset"]
        subset = task_cfg.get("subset")

        if subset:
            _fds_cache[cache_key] = FederatedDataset(
                dataset=dataset_name,
                subset=subset,
                partitioners={"train": partitioner},
            )
        else:
            _fds_cache[cache_key] = FederatedDataset(
                dataset=dataset_name,
                partitioners={"train": partitioner},
            )

    fds = _fds_cache[cache_key]
    partition = fds.load_partition(partition_id)

    # Stratified train/test split (preserve label distribution)
    # Handle rare labels (count < 2): put them in train only, stratify the rest
    from collections import Counter
    from datasets import concatenate_datasets

    label_col = task_cfg["label_field"]
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
    text_fields = task_cfg["text_fields"]

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
    columns_to_remove = [c for c in partition_split["train"].column_names
                         if c not in ("input_ids", "attention_mask", "token_type_ids", "label", "labels")]
    partition_split = partition_split.remove_columns(columns_to_remove)

    # Rename label to labels if needed
    if "label" in partition_split["train"].column_names:
        partition_split = partition_split.rename_column("label", "labels")

    partition_split.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return partition_split["train"], partition_split["test"], tokenizer, data_collator


def load_centralized_data(
    task_name: str,
    model_name: str,
    max_seq_length: int = 128,
):
    """Load full GLUE dataset for centralized training/evaluation.

    Returns:
        (train_dataset, eval_dataset, tokenizer, data_collator): HF Datasets + tokenizer + collator.
    """
    from datasets import load_dataset

    task_cfg = GLUE_TASK_CONFIG[task_name]
    text_fields = task_cfg["text_fields"]

    subset = task_cfg.get("subset")
    if subset:
        dataset = load_dataset(task_cfg["dataset"], subset)
    else:
        dataset = load_dataset(task_cfg["dataset"])

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

    columns_to_remove = [c for c in dataset["train"].column_names
                         if c not in ("input_ids", "attention_mask", "token_type_ids", "label", "labels")]
    dataset = dataset.remove_columns(columns_to_remove)

    if "label" in dataset["train"].column_names:
        dataset = dataset.rename_column("label", "labels")

    dataset.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Use validation set for test
    test_split = "validation_matched" if task_name == "mnli" else "validation"

    return dataset["train"], dataset[test_split], tokenizer, data_collator
