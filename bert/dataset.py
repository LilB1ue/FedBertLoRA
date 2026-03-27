"""GLUE dataset loading with Dirichlet non-IID partitioning for federated learning."""

import warnings

from torch.utils.data import DataLoader
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
    batch_size: int = 32,
    test_size: float = 0.2,
):
    """Load a partition of GLUE data for federated learning.

    Args:
        partition_id: Client partition index.
        num_partitions: Total number of partitions.
        task_name: GLUE task name (sst2, qnli, mnli, qqp, rte).
        model_name: HuggingFace model name for tokenizer.
        dirichlet_alpha: Dirichlet distribution alpha for non-IID split.
        max_seq_length: Maximum token sequence length.
        batch_size: DataLoader batch size.
        test_size: Fraction of data for local validation.

    Returns:
        (trainloader, testloader): PyTorch DataLoaders.
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    task_cfg = GLUE_TASK_CONFIG[task_name]
    cache_key = f"{task_name}_{num_partitions}_{dirichlet_alpha}"

    global _fds_cache
    if cache_key not in _fds_cache:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=task_cfg["label_field"],
            alpha=dirichlet_alpha,
            min_partition_size=10,
            self_balancing=True,
            seed=42,
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

    # Train/test split
    partition_split = partition.train_test_split(test_size=test_size, seed=42)

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

    trainloader = DataLoader(
        partition_split["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    testloader = DataLoader(
        partition_split["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return trainloader, testloader


def load_centralized_data(
    task_name: str,
    model_name: str,
    max_seq_length: int = 128,
    batch_size: int = 32,
):
    """Load full GLUE dataset for centralized training/evaluation.

    Returns:
        (trainloader, testloader): PyTorch DataLoaders.
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainloader = DataLoader(
        dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    # Use validation set for test
    test_split = "validation_matched" if task_name == "mnli" else "validation"
    testloader = DataLoader(
        dataset[test_split],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return trainloader, testloader
