"""
This file defines classes/functions for comparing the MIA's predictions down to sample level.
"""
import os
import pickle
import sys


from typing import Optional
import numpy as np
from torch.utils.data import Dataset
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
from miae.eval_methods.prediction import Predictions


def save_accuracy(pred: List[Predictions], file_path: str, target_fpr: Optional[float] = None):
    """
    save the accuracy of the prediction to a .txt file
    :param pred: the prediction object
    :param file: the file to save the accuracy
    """
    try:
        with open(file_path, "w") as file:
            for p in pred:
                file.write(f"{p.name}\n")
                file.write(f"Accuracy (with threshold): {p.accuracy():.4f}\n")
                # Write the accuracy of the prediction using FPR
                file.write(f"Accuracy (with FPR = {target_fpr}): {p.accuracy_at_fpr(target_fpr):.4f}\n\n")
    except IOError as e:
        print(f"Error: Unable to write to file '{file_path}': {e}")
    finally:
        file.close()


def load_predictions(file_path: str) -> np.ndarray:
    """
    Load predictions from a file.

    :param file_path: path to the file containing the predictions
    :return: prediction as a numpy array
    """
    prediction = np.load(file_path)
    return prediction


def load_target_dataset(filepath: str):
    """
    Load the target dataset (the dataset that's used to test the performance of the MIA)

    :param filepath: path to the files ("index_to_data.pkl" and "attack_set_membership.npy")
    :return: index_to_data (dictionary representing the mapping from index to data), attack_set_membership
    """

    # check if those 2 files exist
    if not os.path.exists(filepath + "/index_to_data.pkl") or not os.path.exists(
            filepath + "/attack_set_membership.npy"):
        raise FileNotFoundError(
            f"The files 'index_to_data.pkl' and 'attack_set_membership.npy' are not found in the given path: \n{filepath}.")

    # Load the dataset
    index_to_data = pickle.load(open(filepath + "/index_to_data.pkl", "rb"))
    attack_set_membership = np.load(filepath + "/attack_set_membership.npy")

    return index_to_data, attack_set_membership


def load_dataset(file_path: str) -> Dataset:
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def pearson_correlation(pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    Calculate the Pearson correlation between two predictions.

    :param pred1: prediction 1 as a numpy array
    :param pred2: prediction 2 as a numpy array
    :return: Pearson correlation between the two predictions
    """
    return np.corrcoef(pred1, pred2)[0, 1]


