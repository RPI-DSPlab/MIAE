"""
This file defines classes/functions for comparing the MIA's predictions down to sample level.
"""
import os
import pickle
import sys

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from experiment import models
from typing import List, Optional, Tuple, Callable, Union
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_unweighted
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms


class Predictions:
    def __init__(self, pred_arr: np.ndarray, ground_truth_arr: np.ndarray, name: str):
        """
        Initialize the Predictions object.

        :param pred_arr: predictions as a numpy array
        :param ground_truth_arr: ground truth as a numpy array
        :param name: name of the attack
        """
        self.pred_arr = pred_arr
        self.ground_truth_arr = ground_truth_arr
        self.name = name

    def predictions_to_labels(self, threshold: float = 0.5) -> np.ndarray:
        """
        Convert predictions to binary labels.

        :param self: Predictions object
        :param threshold: threshold for converting predictions to binary labels
        :return: binary labels as a numpy array
        """
        labels = (self.pred_arr > threshold).astype(int)
        return labels

    def accuracy(self) -> float:
        """
        Calculate the accuracy of the predictions.

        :param self: Predictions object
        :return: accuracy of the predictions
        """
        return np.mean(self.predictions_to_labels() == self.ground_truth_arr)

    def difference_pred_gt(self):
        """
        Calculate the difference between the predictions and the ground truth.

        :param self: Predictions object
        :return: difference between the predictions and the ground truth
        """
        return self.pred_arr - self.ground_truth_arr



def plot_venn_diagram(pred_list: List[Predictions], title: str, save_path: str):
    """
    Plot the Venn diagram for the predictions from different attacks.

    :param pred_list: list of Predictions from different attacks
    :param title: title of the graph
    :param save_path: path to save the graph
    """
    if len(pred_list) < 2:
        raise ValueError("At least 2 attacks are required for comparison.")

    # Create a dictionary to store attacked points for each implementation
    attacked_points = {pred.name: set() for pred in pred_list}

    # Populate the attacked_points dictionary
    for pred in pred_list:
        attacked_points[pred.name] = set(np.where(pred.predictions_to_labels() == pred.ground_truth_arr)[0])

    # Plot the Venn diagram
    plt.figure(figsize=(7, 7), dpi=300)  # MUST HAVE
    circle_colors = ['red', 'blue', 'green', 'purple', 'orange']
    venn_sets = [attacked_points[pred.name] for pred in pred_list]
    venn_labels = [pred.name for pred in pred_list]
    venn2(venn_sets, set_labels=venn_labels, set_colors=circle_colors)
    plt.title(title)
    plt.savefig(save_path, dpi=300)


def plot_t_sne(pred_list: List[Predictions], title: str, save_path: str, perplexity: int = 30):
    """
    Plot the t-SNE graph for the predictions from different attacks.

    :param pred_list: list of Predictions from different attacks
    :param title: title of the graph
    :param save_path: path to save the graph
    :param perplexity: perplexity for t-SNE (default: 30)
    """
    # check if the number of attacks is less than 2
    if len(pred_list) < 2:
        raise ValueError("At least 2 attacks are required for comparison.")

    # load ResNet56 model


def load_image_by_index(dataset: ConcatDataset, index_list: np.ndarray, title: str, save_path: str):
    """
    Load and plot the image by index from the dataset
    :param dataset: the dataset
    :param index_list: list of indices
    :param title: title of the graph
    :param save_path: path to save the graph
    """
    # Define the unnormalization transformation
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
    ])

    # Create a subplot for each image in the index_list
    num_images = len(index_list)
    num_rows = 3
    num_cols = num_images // num_rows
    plt.figure(figsize=(num_cols * 3, num_rows * 3), dpi=300)
    plt.axis('off')
    plt.title(title)
    for i, index in enumerate(index_list, 1):
        img, _ = dataset[index]
        img_unnormalized = unnormalize(img)
        img_np = img_unnormalized.permute(1, 2, 0).numpy()
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(img_np)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)


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
        fpr, tpr, _ = roc_curve(x, score)
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


def averaging_predictions(pred_list: List[Predictions]) -> np.ndarray:
    """
    Average the predictions from different attacks.

    :param pred_list: list of Predictions
    :return: averaged prediction
    """
    pred_list = [pred.pred_arr for pred in pred_list]
    return np.mean(pred_list, axis=0)

def summation_predictions(pred_list: List[Predictions]) -> np.ndarray:
    """
    Sum the predictions from different attacks.

    :param pred_list: list of Predictions
    :return: summed prediction
    """
    pred_list = [pred.pred_arr for pred in pred_list]
    return np.sum(pred_list, axis=0)


def majority_voting(pred_list: List[Predictions]) -> np.ndarray:
    """
    Majority voting for the predictions from different attacks.

    :param pred_list: list of Predictions
    :return: majority voted prediction
    """
    # convert predictions to binary labels
    labels_list = [pred.predictions_to_labels(threshold=0.5) for pred in pred_list]

    # calculate the majority voted prediction
    majority_voted_labels = np.mean(labels_list, axis=0)
    majority_voted_labels = (majority_voted_labels > 0.5).astype(int)
    return majority_voted_labels


def unanimous_voting(pred_list: List[Predictions]) -> np.ndarray:
    """
    Unanimous voting for the predictions from different attacks.

    :param pred_list: list of predictions
    :return: unanimous voted prediction
    """
    # convert predictions to binary labels
    labels_list = [pred.predictions_to_labels(threshold=0.5) for pred in pred_list]

    # calculate the unanimous voted prediction
    unanimous_voted_labels = np.mean(labels_list, axis=0)
    unanimous_voted_labels = (unanimous_voted_labels == 1).astype(int)
    return unanimous_voted_labels


# -------- below are the functions for comparing with example hardness --------

def load_example_hardness(file_path: str) -> np.ndarray:
    """
    Load the example hardness from a file and parse it to a numpy int array

    :param file_path: path to the file containing the example hardness
    :return: example hardness as a numpy array
    """
    with open(file_path, "rb") as f:
        example_hardness = pickle.load(f)

    # parse the example hardness to a numpy int array
    example_hardness = np.array(example_hardness, dtype=int)
    return example_hardness


def plot_example_hardness_dis(eh: np.ndarray, save_path: str, title: str):
    """
    Plot the example hardness distribution.

    :param eh: example hardness as a numpy array
    :param save_path: path to save the graph
    :param title: title of the graph
    """
    print(eh)
    num_bin = max(eh)
    plt.figure(figsize=(6, 5))
    plt.hist(eh, bins=num_bin, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Example Hardness")
    plt.ylabel("Frequency")
    plt.savefig(save_path, dpi=300)


def plot_example_hardness_vs_diff_pred_gt(eh: np.ndarray, predictions: Predictions, save_path: str, title: str):
    """
    Plot the example hardness vs the difference between the predictions and the ground truth.

    :param eh: example hardness as a numpy array
    :param predictions: Predictions object
    :param save_path: path to save the graph
    :param title: title of the graph
    """
    diff_pred_gt = predictions.difference_pred_gt()
    plt.figure(figsize=(6, 5))
    plt.scatter(eh, diff_pred_gt, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel("Example Hardness")
    plt.ylabel("Difference between the Predictions and the Ground Truth")
    plt.savefig(save_path, dpi=300)

