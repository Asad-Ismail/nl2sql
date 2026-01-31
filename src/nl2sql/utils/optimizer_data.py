"""Unified data loading for all optimizers."""

from datasets import load_dataset
from nl2sql.utils.util import load_schemas

SCHEMAS = load_schemas()


def load_optimizer_data(
    dataset_name: str = "AsadIsmail/nl2sql-deduplicated",
    train_file: str = "spider_clean.jsonl",
    dev_file: str = "spider_dev_clean.jsonl",
    train_size: int = 400,
    val_size: int = 500,
    shuffle_seed: int = 42,
    format: str = "dict",
) -> tuple:
    """
    Load and split data for optimizer training/evaluation.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset name
    train_file : str
        Training data file
    dev_file : str
        Development data file
    train_size : int
        Number of training examples
    val_size : int
        Number of validation examples
    shuffle_seed : int
        Random seed for shuffling
    format : str
        Output format: "dict" or "dspy"

    Returns
    -------
    tuple
        (train_data, val_data, dev_data)
        Format depends on `format` parameter:
        - "dict": Lists of dictionaries
        - "dspy": Lists of dspy.Example objects
    """
    # Load datasets
    full_data = load_dataset(
        dataset_name,
        data_files=train_file,
        split="train",
    )
    dev_data = load_dataset(
        dataset_name,
        data_files=dev_file,
        split="train",
    )

    # Shuffle
    shuffled = full_data.shuffle(seed=shuffle_seed)

    # Split
    train = shuffled.select(range(0, train_size))
    val = shuffled.select(range(train_size, train_size + val_size))
    dev = dev_data

    # Convert format
    if format == "dspy":
        import dspy

        train = [_to_dspy_example(ex) for ex in train]
        val = [_to_dspy_example(ex) for ex in val]
        dev = [_to_dspy_example(ex) for ex in dev]
    else:
        train = list(train)
        val = list(val)
        dev = list(dev)

    return train, val, dev


def _to_dspy_example(example):
    """Convert dict to dspy.Example."""
    import dspy

    return dspy.Example(
        db_schema=SCHEMAS.get(example["db_id"], f"Database: {example['db_id']}"),
        question=example["question"],
        sql=example["sql"],
        db_id=example["db_id"],
    ).with_inputs("db_schema", "question")
