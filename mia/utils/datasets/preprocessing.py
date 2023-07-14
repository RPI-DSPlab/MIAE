import torch
import numpy as np

from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms


def apply_global_contrast_normalization(input_tensor: torch.Tensor, normalization_scale: str = 'l2') -> torch.Tensor:
    """
    Apply global contrast normalization to an input tensor. The function subtracts the mean across features (pixels)
    and normalizes by the specified scale. The scale can either be the standard deviation ('l2') or L1-norm ('l1').
    Note that this is a per sample normalization globally across features and not across the entire datasets.

    Args:
        input_tensor (torch.Tensor): The input tensor to be normalized.
        normalization_scale (str, optional): The type of normalization to apply. Can be either 'l1' or 'l2'.
                                            Defaults to 'l2'.

    Returns:
        torch.Tensor: The normalized tensor.

    Raises:
        AssertionError: If the normalization_scale is not 'l1' or 'l2'.
    """
    assert normalization_scale in ('l1', 'l2'), "Invalid normalization scale. Choose either 'l1' or 'l2'."

    total_features = int(np.prod(input_tensor.shape))

    feature_mean = torch.mean(input_tensor)  # mean over all features (pixels) per sample
    input_tensor -= feature_mean
    scale_value = None

    if normalization_scale == 'l1':
        scale_value = torch.mean(torch.abs(input_tensor))

    if normalization_scale == 'l2':
        scale_value = torch.sqrt(torch.sum(input_tensor ** 2)) / total_features

    input_tensor /= scale_value

    return input_tensor


def find_target_label_indices(label_array: np.ndarray, target_labels) -> list:
    """
    Identify the indices of labels present in the given target labels.

    Args:
        label_array (np.ndarray): An array containing labels.
        target_labels (list/tuple): A list or tuple of target labels.

    Returns:
        list: A list containing the indices of target labels in the label array.
    """
    target_indices = np.argwhere(np.isin(label_array, target_labels)).flatten().tolist()
    return target_indices

