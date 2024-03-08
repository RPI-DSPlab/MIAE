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
from matplotlib_venn import venn2, venn2_unweighted, venn3
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


def plot_venn_diagram(pred_list: List[Predictions], title: str, save_path: str, threshold: float = 0.5, goal: str = "attack_compare"):
    """
    Plot the Venn diagram for the predictions from different attacks or for one attack but with different seed.

    :param pred_list: list of Predictions from different attacks
    :param title: title of the graph
    :param save_path: path to save the graph
    :param threshold: threshold for considering a point as attacked
    :param goal: goal of the comparison ("attack_compare" or "seed_compare")
    """
    if len(pred_list) < 2 and goal == "attack_compare":
        raise ValueError("At least 2 attacks are required for comparison.")
    elif len(pred_list) < 2 and goal == "seed_compare":
        raise ValueError("At least from 2 different seeds are required for comparison.")

    # get the attacked points
    attacked_points = {pred.name: set() for pred in pred_list}
    plt.figure(figsize=(7, 7), dpi=300)
    if goal == "attack_compare":
        for pred in pred_list:
            attacked_points[pred.name] = set(np.where((pred.predictions_to_labels() == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets = [attacked_points[pred.name] for pred in pred_list]
        venn_function = venn2
    elif goal == "seed_compare":
        for pred in pred_list:
            attacked_points[pred.name] = set(np.where((pred.predictions_to_labels() == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets = tuple(attacked_points[pred.name] for pred in pred_list)
        venn_function = venn3

    circle_colors = ['red', 'blue', 'green', 'purple', 'orange']
    venn_labels = [pred.name for pred in pred_list]
    venn_function(subsets=venn_sets, set_labels=venn_labels, set_colors=circle_colors)
    plt.title(title)
    plt.savefig(save_path, dpi=300)


def plot_t_sne(pred_list: List[Predictions], dataset: ConcatDataset, title: str, save_path: str, perplexity: int = 30):
    """
    Plot the t-SNE graph for the predictions from different attacks.

    :param pred_list: list of Predictions from different attacks
    :param dataset: the dataset
    :param title: title of the graph
    :param save_path: path to save the graph
    :param perplexity: perplexity for t-SNE (default: 30)
    """
    # check if the number of attacks is less than 2
    if len(pred_list) < 2:
        raise ValueError("At least 2 attacks are required for comparison.")

    # get the high-dimensional data
    high_dim_data = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        high_dim_data.append(img.numpy().flatten())
    high_dim_data = np.array(high_dim_data)
    print(f"the shape of the high_dim_data: {high_dim_data.shape} and the 1st image: {high_dim_data[0]}")

    # get the low-dimensional data by applying t-SNE
    tsne = TSNE(n_components=2, perplexity=20)
    low_dim_data = tsne.fit_transform(high_dim_data)
    print(f"the shape of the low_dim_data: {low_dim_data.shape} and the 1st image: {low_dim_data[0]}")

    # Separate the data into four categories
    # Create indices for different attack categories
    indices_by_attack1 = np.where(pred_list[0].predictions_to_labels() == pred_list[0].ground_truth_arr)[0]
    indices_by_attack2 = np.where(pred_list[1].predictions_to_labels() == pred_list[1].ground_truth_arr)[0]
    indices_by_both = np.intersect1d(indices_by_attack1, indices_by_attack2)
    indices_by_none = np.setdiff1d(np.arange(len(dataset)), np.union1d(indices_by_attack1, indices_by_attack2))
    # print the number of points in each category
    print(f"Number of points in {pred_list[0].name}: {len(indices_by_attack1)}")
    print(f"Number of points in {pred_list[1].name}: {len(indices_by_attack2)}")
    print(f"Number of points in both: {len(indices_by_both)}")
    print(f"Number of points in none: {len(indices_by_none)}")

    # plot the t-SNE graph based on the low-dimensional data
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(low_dim_data[indices_by_attack1, 0], low_dim_data[indices_by_attack1, 1], label=pred_list[0].name, c='r', s=1)
    plt.scatter(low_dim_data[indices_by_attack2, 0], low_dim_data[indices_by_attack2, 1], label=pred_list[1].name, c='b', s=1)
    plt.scatter(low_dim_data[indices_by_both, 0], low_dim_data[indices_by_both, 1], label="Both", c='g', s=1)
    plt.scatter(low_dim_data[indices_by_none, 0], low_dim_data[indices_by_none, 1], label="None", c='k', s=1)

    # Set limits for the plot based on the range of low-dimensional data
    min_x, min_y = np.min(low_dim_data, axis=0)
    max_x, max_y = np.max(low_dim_data, axis=0)
    margin = 5  # Add some margin for better visualization

    plt.xlim(min_x - margin, max_x + margin)
    plt.ylim(min_y - margin, max_y + margin)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300)



def plot_image_by_index(dataset: ConcatDataset, index_list: np.ndarray, title: str, save_path: str):
    """
    plot the image by index from the data set
    :param dataset: the data set
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
