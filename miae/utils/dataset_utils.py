import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List
from copy import deepcopy
from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets


def get_xy_from_dataset(dataset: Dataset) -> (np.ndarray, np.ndarray):
    """
    Get x and y from a dataset
    :param dataset: dataset
    :return: x and y
    """
    x = []
    y = []

    for item in dataset:
        data, label = item
        x.append(data.numpy())
        y.append(label)

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)

    return x, y


def get_num_classes(dataset: torch.utils.data.TensorDataset) -> int:
    """
    Get the number of classes in a dataset
    :param dataset: dataset
    :return: number of classes
    """
    labels = [label for _, label in dataset]
    unique_classes = set(labels)
    return len(unique_classes)


def dataset_split(dataset, lengths: list, shuffle_seed=1):
    """
    Split the dataset into subsets.
    :param dataset: the dataset.
    :param lengths: the lengths of each subset.
    :param shuffle_seed: the seed for shuffling the dataset.
    """
    np.random.seed(shuffle_seed)
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.shuffle(indices)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(torch._utils._accumulate(lengths), lengths)]

def load_mimir_dataset(name: str, split: str, cache_dir: str, test_size: float, seed: int) -> DatasetDict:
    """
    Load the MIMIR dataset and split it into training and testing sets separately
    for member and non-member data.

    :param name: The name of the dataset such as "arxiv", "pile_cc", etc.
    :param split: The split of the dataset such as "ngram_13_0.8".
    :param cache_dir: Directory to cache the dataset.
    :param test_size: Proportion of the dataset to include in the test split.
    :param seed: Random seed for reproducibility.
    :return: A DatasetDict with 'train' and 'test' splits.
    """
    # Load the dataset
    dataset = load_dataset("iamgroot42/mimir", name, split=split, cache_dir=cache_dir)

    if "member" not in dataset.column_names:
        raise ValueError("The dataset does not contain the column 'member'")
    if "nonmember" not in dataset.column_names:
        raise ValueError("The dataset does not contain the column 'non-member'")

    # Split the member and non-member datasets separately
    member_texts = dataset["member"]
    nonmember_texts = dataset["nonmember"]

    member_dataset = Dataset.from_dict({"text": member_texts, "label": [1] * len(member_texts)})
    nonmember_dataset = Dataset.from_dict({"text": nonmember_texts, "label": [0] * len(nonmember_texts)})

    member_split = member_dataset.train_test_split(test_size=test_size, seed=seed)
    nonmember_split = nonmember_dataset.train_test_split(test_size=test_size, seed=seed)

    # Combine the member and non-member splits
    # Use concatenate_datasets to combine the member and non-member splits
    train_texts = concatenate_datasets([member_split["train"], nonmember_split["train"]])
    test_texts = concatenate_datasets([member_split["test"], nonmember_split["test"]])

    # Create a new DatasetDict with the combined train and test sets
    split_dataset = DatasetDict({"train": train_texts, "test": test_texts})

    return split_dataset
