import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class CanarySet(Dataset):
    def __init__(self, subset, canary_map):
        """
        Initialize the CanarySett.
        
        Args:
            subset (torch.utils.data.Subset): The original Subset dataset.
            canary_map (dict): A dictionary mapping canary indices to their new labels.
        """
        self.subset = subset
        self.canary_map = canary_map  # {index: canary_label}

    def __getitem__(self, index):
        """
        Get an item from the dataset, overriding the label if it's a canary index.
        
        Args:
            index (int): Index of the item.
        
        Returns:
            tuple: (features, label), where label is modified if index is a canary.
        """
        data = self.subset[index]  # Get (features, original_label)
        if index in self.canary_map:
            features, _ = data
            return (features, self.canary_map[index])
        return data

    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.subset)

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


def add_canaries(dataset, num_canaries, num_classes, shuffle_seed=1):
    """
    Add canaries to the dataset by creating a new dataset with modified labels.
    
    Args:
        dataset (torch.utils.data.Subset): The input dataset (a Subset).
        num_canaries (int): Number of canaries to add.
        num_classes (int): Number of possible classes for labels.
        shuffle_seed (int): Seed for random shuffling of indices.
    
    Returns:
        tuple: (new_dataset, canary_indices)
            - new_dataset (CanarySubset): Dataset with canaries added.
            - canary_indices (np.ndarray): Indices where canaries were added.
    """
    np.random.seed(shuffle_seed)
    canary_indices = np.random.choice(len(dataset), num_canaries, replace=False)
    
    # Create a mapping of canary indices to new labels
    canary_map = {}
    for idx in canary_indices:
        original_label = dataset[idx][1]  # Get the original label
        # Choose a random label different from the original
        possible_labels = [label for label in range(num_classes) if label != original_label]
        canary_label = np.random.choice(possible_labels)
        canary_map[idx] = canary_label
    
    # Return the new dataset and the canary indices
    return CanarySet(dataset, canary_map), canary_indices

