import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

def get_xy_from_dataset(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
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
    labels = [int(label) for _, label in dataset]
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

