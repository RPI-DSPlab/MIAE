"""
This file defines classes/functions for comparing the MIA's predictions down to sample level.
"""
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


def load_predictions(file_path: str) -> np.ndarray:
    """
    Load predictions from a file.

    :param file_path: path to the file containing the predictions
    :return: prediction as a numpy array
    """
    prediction = np.load(file_path)
    return prediction


def predictions_to_labels(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert predictions to binary labels.

    :param predictions: predictions as a numpy array
    :param threshold: threshold for converting predictions to binary labels
    :return: binary labels as a numpy array
    """
    labels = (predictions > threshold).astype(int)
    return labels


def load_target_dataset(filepath: str):
    """
    Load the target dataset (the dataset that's used to test the performance of the MIA)

    :param filepath: path to the files ("index_to_data.pkl" and "attack_set_membership.npy")
    :return: index_to_data (dictionary representing the mapping from index to data), attack_set_membership
    """

    # Load the dataset
    index_to_data = pickle.load(open(filepath + "index_to_data.pkl", "rb"))
    attack_set_membership = np.load(filepath + "attack_set_membership.npy")

    return index_to_data, attack_set_membership


def plot_auc_graph(pred_list: list[np.ndarray],
                   name_list: list[str],
                   ground_truth_arr: np.ndarray,
                   title: str, save_path: str = None
                   ):
    """
    plot the AUC graph for the predictions from different attacks
    :param pred_list: np.ndarray list of predictions
    :param name_list: list of names for the attacks
    :param ground_truth_arr: np.ndarray of ground truth
    :param title: title of the graph
    :param save_path: path to save the graph
    """

    plt.figure(figsize=(10, 6))

    for preds, name in zip(pred_list, name_list):
        auc_score = roc_auc_score(ground_truth_arr, preds)
        fpr, tpr, _ = roc_curve(ground_truth_arr, preds)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # prints the auc, TPR@FPR=0.01, max accuracy
    for preds, name in zip(pred_list, name_list):
        fpr, tpr, _ = roc_curve(ground_truth_arr, preds)
        print(f"{name}: AUC = {roc_auc_score(ground_truth_arr, preds):.2f}, TPR@FPR=0.01 = {tpr[np.argmin(np.abs(fpr - 0.01))]:.2f}, max accuracy = {max(tpr - fpr):.2f}")


def pearson_correlation(pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    Calculate the Pearson correlation between two predictions.

    :param pred1: prediction 1 as a numpy array
    :param pred2: prediction 2 as a numpy array
    :return: Pearson correlation between the two predictions
    """
    return np.corrcoef(pred1, pred2)[0, 1]


def averaging_predictions(pred_list: list[np.ndarray]) -> np.ndarray:
    """
    Average the predictions from different attacks.

    :param pred_list: list of predictions
    :return: averaged prediction
    """
    return np.mean(pred_list, axis=0)


def majority_voting(pred_list: list[np.ndarray]) -> np.ndarray:
    """
    Majority voting for the predictions from different attacks.

    :param pred_list: list of predictions
    :return: majority voted prediction
    """
    # convert predictions to binary labels
    labels_list = [predictions_to_labels(pred, threshold=0.5) for pred in pred_list]

    # calculate the majority voted prediction
    majority_voted_labels = np.mean(labels_list, axis=0)
    majority_voted_labels = (majority_voted_labels > 0.5).astype(int)
    return majority_voted_labels

