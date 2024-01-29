import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List
from copy import deepcopy


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
        x.append(data)
        y.append(label)

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)

    return x, y
