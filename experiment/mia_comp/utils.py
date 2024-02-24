"""
This file defines classes/functions for comparing the MIA's predictions down to sample level.
"""
import os
import pickle
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Callable, Union
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np


class Predictions:
    def __init__(self, pred_arr: np.ndarray, ground_truth_arr: np.ndarray, name: str):
        """
        Initialize the Predictions object.

        :param pred_arr: predictions as a numpy array
        :param name: name of the attack
        """
        self.pred_arr = pred_arr
        self.ground_truth_arr = ground_truth_arr
        self.name = name


def plot_venn_diagram(pred_list: List[Predictions], save_path: str):
    """
    Plot the Venn diagram for the predictions from different attacks.

    :param pred_list: list of Predictions from different attacks
    """

    # TODO for Chengyu: Implement this function
    pass


def plot_t_sne(pred_list: List[Predictions], save_path: str, perplexity: int = 30):
    """
    Plot the t-SNE graph for the predictions from different attacks.

    :param pred_list: list of Predictions from different attacks
    :param save_path: path to save the graph
    :param perplexity: perplexity for t-SNE (default: 30)
    """

    # TODO for Chengyu: Implement this function
    pass



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
    labels = (predictions < threshold).astype(int)
    return labels


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


def accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate the accuracy of the predictions.

    :param predictions: predictions as a numpy array
    :param ground_truth: ground truth as a numpy array
    :return: accuracy of the predictions
    """
    return np.mean(predictions_to_labels(predictions) == ground_truth)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_auc_graph(pred_list: list[np.ndarray],
                   name_list: list[str],
                   ground_truth_arr: np.ndarray,
                   title: str, save_path: str = None,
                   log_scale: bool = True
                   ):
    """
    !! This function is deprecated. Use custom_auc instead. !!
    plot the AUC graph for the predictions from different attacks
    :param pred_list: np.ndarray list of predictions
    :param name_list: list of names for the attacks
    :param ground_truth_arr: np.ndarray of ground truth
    :param title: title of the graph
    :param save_path: path to save the graph
    :param log_scale: whether to use log scale for both axes (default: False)
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

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # prints the auc, TPR@FPR=0.01, max accuracy
    for preds, name in zip(pred_list, name_list):
        fpr, tpr, _ = roc_curve(ground_truth_arr, preds)
        print(
            f"{name}: AUC = {roc_auc_score(ground_truth_arr, preds):.2f}, TPR@FPR=0.01 = {tpr[np.argmin(np.abs(fpr - 0.01))]:.2f}, max accuracy = {max(tpr - fpr):.2f}")


def custom_auc(pred_list: List[np.ndarray],
               name_list: list[str],
               ground_truth_arr: np.ndarray,
               title: str, save_path: str = None
               ):
    """
    plot the AUC graph for the predictions from different attacks (ported from Yuetian's code)
    :param pred_list: np.ndarray list of predictions
    :param name_list: list of names for the attacks
    :param ground_truth_arr: np.ndarray of ground truth
    :param title: title of the graph
    :param save_path: path to save the graph
    """

    def sweep(score: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute a Receiver Operating Characteristic (ROC) curve.

        Args:
            score (np.ndarray): The predicted scores.
            x (np.ndarray): The ground truth labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, float]: The False Positive Rate (FPR),
            True Positive Rate (TPR), Area Under the Curve (AUC), and Accuracy.
        """
        fpr, tpr, _ = roc_curve(x, -score)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        return fpr, tpr, auc(fpr, tpr), acc

    def do_plot(prediction: np.ndarray,
                answers: np.ndarray,
                legend: str = '',
                sweep_fn: Callable = sweep,
                **plot_kwargs: Union[int, str, float]) -> Tuple[float, float]:
        """
        Generate the ROC curves.

        Args:
            prediction (np.ndarray): The predicted scores.
            answers (np.ndarray): The ground truth labels.
            legend (str, optional): Legend for the plot. Defaults to ''.
            sweep_fn (Callable, optional): Function used to compute the ROC curve. Defaults to sweep.

        Returns:
            Tuple[float, float]: Accuracy and Area Under the Curve (AUC).
        """
        fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

        low = tpr[np.where(fpr < .001)[0][-1]] if np.any(fpr < .001) else 0

        print(f'Attack: {legend.strip():<20} AUC: {auc:<8.4f} max Accuracy: {acc:<8.4f} TPR@0.1%FPR: {low:<8.4f}')

        metric_text = f'auc={auc:.3f}'

        plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)

        return acc, auc

    plt.figure(figsize=(6, 5))
    plt.title(title)

    membership_list = [ground_truth_arr for _ in range(len(name_list))]
    for prediction, answer, legend in zip(pred_list, membership_list, name_list):
        do_plot(prediction, answer,
                f"{legend}\n")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    # plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


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


def unanimous_voting(pred_list: list[np.ndarray]) -> np.ndarray:
    """
    Unanimous voting for the predictions from different attacks.

    :param pred_list: list of predictions
    :return: unanimous voted prediction
    """
    # convert predictions to binary labels
    labels_list = [predictions_to_labels(pred, threshold=0.5) for pred in pred_list]

    # calculate the unanimous voted prediction
    unanimous_voted_labels = np.mean(labels_list, axis=0)
    unanimous_voted_labels = (unanimous_voted_labels == 1).astype(int)
    return unanimous_voted_labels
