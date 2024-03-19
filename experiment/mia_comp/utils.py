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
from matplotlib_venn import venn2, venn2_unweighted, venn3, venn3_unweighted
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms
from typing import Dict, List, Tuple



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

    def compute_fpr(self):
        """
        Compute the false positive rate (FPR) of the predictions.
        """
        # Convert predictions and ground truth to PyTorch tensors if they are not already
        pred_tensor = torch.tensor(self.pred_arr)
        ground_truth_tensor = torch.tensor(self.ground_truth_arr)
        false_positive = torch.logical_and(pred_tensor == 1, ground_truth_tensor == 0).sum().item()
        true_negative = torch.logical_and(pred_tensor == 0, ground_truth_tensor == 0).sum().item()
        total_negative = true_negative + false_positive
        FPR = false_positive / total_negative if total_negative > 0 else 0
        return FPR

    def adjust_fpr(self, target_fpr):
        """
        Adjust the predictions to achieve a target FPR.
        :param target_fpr: target FPR
        :return: adjusted predictions as a numpy array
        """
        pred_tensor = torch.tensor(self.pred_arr).float()

        current_fpr = self.compute_fpr()
        if current_fpr < target_fpr:
            adjusted_pred_arr = self.pred_arr.copy()
            return adjusted_pred_arr

        threshold = torch.quantile(pred_tensor, 1 - target_fpr)
        adjusted_pred_arr = (pred_tensor >= threshold).float().numpy()
        return adjusted_pred_arr

    def get_tp(self):
        """
        Get the indices of the true positive samples.
        """
        return np.where((self.predictions_to_labels() == 1) & (self.ground_truth_arr == 1))[0]

class SampleHardness:
    def __init__(self, score_arr, name: str):
        """

        """
        self.score_arr = score_arr
        self.name = name

        self.max_score = int(np.max(self.score_arr))
        self.min_score = int(np.min(self.score_arr))

    def plot_distribution(self, save_dir):
        """
        plot a histogram of the sample hardness scores
        """
        plt.hist(self.score_arr, bins=20, range=(self.min_score, self.max_score))
        plt.title(f"Distribution of {self.name}")
        plt.xlabel(f"{self.name}")
        plt.ylabel("Number of Samples")
        plt.savefig(save_dir)

    def plot_distribution_pred_TP(self, tp, save_path, title=None):
        """
        Plot the distribution of the sample hardness scores for the true positive samples and entire score_arr.

        :param tp: true positive samples
        :param score_arr: sample hardness scores
        """
        # Define colors for each distribution
        plt.clf()
        all_samples_color = 'blue'
        true_positives_color = 'orange'

        # Plot histogram for all samples
        plt.hist(self.score_arr, bins=self.max_score, range=(np.min(self.score_arr), np.max(self.score_arr)),
                 color=all_samples_color, alpha=0.5, label='All Samples')

        # Plot histogram for true positive samples
        plt.hist(self.score_arr[list(tp)], bins=self.max_score, range=(np.min(self.score_arr), np.max(self.score_arr)),
                 color=true_positives_color, alpha=0.5, label='TP Sample')

        # Set plot title and labels
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Distribution of {self.name} for True Positives and All Samples")
        plt.xlabel(f"{self.name}")
        plt.ylabel("Number of Samples")

        # Add legend
        plt.legend()

        # Save the plot
        plt.savefig(save_path, dpi=300)


def common_tp(preds: List[Predictions], fpr=None):
    """
    Find the common true positive samples among the predictions
    Note that this is used for both different attacks or same attack with different seeds.

    :param preds: list of Predictions
    :param fpr: FPR values for adjusting the predictions
    """
    if fpr is None:
        TP = [np.where((pred.predictions_to_labels() == 1) & (pred.ground_truth_arr == 1))[0] for pred in preds]
    else:
        adjusted_preds = [pred.adjust_fpr(fpr) for pred in preds]
        TP = [np.where((adjusted_preds[i] == 1) & (preds[i].ground_truth_arr == 1))[0] for i in range(len(preds))]
    common_TP = set(TP[0])
    for i in range(1, len(TP)):
        common_TP = common_TP.intersection(set(TP[i]))
    return common_TP

def common_tp_preds(pred_list: List[Predictions]) -> Predictions:
    """
    Get the common true positive predictions across different seeds of a single attack

    :param pred_list: List of Predictions objects for the same attack but different seeds
    :return: Predictions object containing only common true positives
    """
    if len(pred_list) < 2:
        raise ValueError("At least 2 predictions are required for comparison.")

    # get the common true positive predictions using logical and
    common_tp = pred_list[0].predictions_to_labels()
    for i in range(1, len(pred_list)):
            common_tp = np.logical_and(common_tp, pred_list[i].predictions_to_labels())

    # create a new Predictions object for the common true positive predictions
    ground_truth_arr = pred_list[0].ground_truth_arr
    name = pred_list[0].name.split()[0]
    return Predictions(common_tp, ground_truth_arr, name)

def data_process_for_venn(pred_dict: Dict[str, List[Predictions]], threshold: Optional[float] = 0, target_fpr: Optional[float] = 0) -> List[Predictions]:
    """
    Process the data for the Venn diagram: get the pred_list
    :param pred_dict: dictionary of Predictions from different attacks, key: attack name, value: list of Predictions of different seeds
    :param threshold: threshold for the comparison (only used when the graph is generated by threshold otherwise None)
    :param target_fpr: target FPR for the comparison (only used when the graph is generated by FPR otherwise None)
    :param name: name of the attack (only used when we want to compare a single attack with different seeds)
    """
    if len(pred_dict) < 2:
        raise ValueError("There is not enough data for comparison.")

    if threshold != 0:
        result = []
        counter = 0
        for attack, pred_obj_list in pred_dict.items():
            counter += 1
            common_tp_pred = common_tp_preds(pred_obj_list)
            result.append(common_tp_pred)
    elif target_fpr != 0:
        adjusted_pred_dict = {}
        result = []
        for attack, pred_obj_list in pred_dict.items():
            adjusted_pred_list = []
            for pred in pred_obj_list:
                adjusted_pred_arr = pred.adjust_fpr(target_fpr)
                adjusted_pred_obj = Predictions(adjusted_pred_arr, pred.ground_truth_arr, pred.name)
                adjusted_pred_list.append(adjusted_pred_obj)
            adjusted_pred_dict[attack] = adjusted_pred_list

        for attack, adjusted_list in adjusted_pred_dict.items():
            common_tp_pred = common_tp_preds(adjusted_list)
            result.append(common_tp_pred)
    else:
        raise ValueError("Either threshold or target_fpr should be provided.")

    return result

def plot_venn_diagram(pred_list: List[Predictions], goal:str, title: str, save_path: str):
    """
    plot the Venn diagram for the predictions based on the goal.
    :param pred_list: list of Predictions from different attacks
    :param title: title of the graph
    :param save_path: path to save the graph
    :param goal: goal of the comparison <== choices: "common_tp", "single_attack"
    """
    # Initialize the attacked points and figure
    attacked_points = {pred.name: set() for pred in pred_list}
    plt.figure(figsize=(8, 8), dpi=300)
    venn_function = venn3_unweighted

    if goal == "common_tp":
        for pred in pred_list:
            attacked_points[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets = [attacked_points[pred.name] for pred in pred_list]
        venn_function = venn2_unweighted if len(pred_list) == 2 else venn3_unweighted
    elif goal == "single_attack":
        for pred in pred_list:
            attacked_points[pred.name] = set(np.where((pred.predictions_to_labels() == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets = tuple(attacked_points[pred.name] for pred in pred_list)
        venn_function = venn2_unweighted if len(pred_list) == 2 else venn3_unweighted


    circle_colors = ['red', 'blue', 'green', 'purple', 'orange']
    venn_labels = [pred.name for pred in pred_list]
    venn_function(subsets=venn_sets, set_labels=venn_labels, set_colors=circle_colors)
    plt.title(title)
    plt.savefig(f"{save_path}.png", dpi=300)

def find_pairwise_preds(pred_list: List[Predictions]) -> List[Tuple[Predictions, Predictions]]:
    """
    Find all possible pairs of predictions in the given list.
    :param pred_list: list of Predictions
    :return: list of tuples, each containing a pair of Predictions
    """
    pairs = []
    n = len(pred_list)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((pred_list[i], pred_list[j]))
    return pairs

def plot_venn_diagram_pairwise(pred_pair_list: List[Tuple[Predictions, Predictions]], graph_title: str, save_path: str):
    """
    Plot Venn diagrams for each pair of predictions in the given list.
    :param pred_pair_list: list of tuples, each containing a pair of Predictions objects
    :param graph_title: title of the graph
    :param save_path: path to save the graphs
    """
    plt.figure(figsize=(8, 8), dpi=300)
    for idx, pair in enumerate(pred_pair_list):
        pred_1, pred_2 = pair
        attacked_points_1 = set(np.where((pred_1.pred_arr == 1) & (pred_1.ground_truth_arr == 1))[0])
        attacked_points_2 = set(np.where((pred_2.pred_arr == 1) & (pred_2.ground_truth_arr == 1))[0])
        circle_colors = ['red', 'blue', 'green', 'purple', 'orange']
        venn2_unweighted(subsets=(attacked_points_1, attacked_points_2), set_labels=(pred_1.name, pred_2.name), set_colors=circle_colors)
        plt.title(f"{graph_title}: {pred_1.name} vs {pred_2.name}")
        plt.savefig(f"{save_path}_{pred_1.name}_vs_{pred_2.name}.png", dpi=300)
        plt.close()


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


def load_example_hardness(file_path: str) -> np.ndarray:
    """
    Load example hardness scores from a file.

    :param file_path: path to the file containing the example hardness scores
    :return: example hardness scores as a numpy array
    """
    with open(file_path, "rb") as f:
        example_hardness = pickle.load(f)
    return example_hardness


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


def plot_auc(pred_list: List[np.ndarray],
             name_list: List[str],
             ground_truth_arr: np.ndarray,
             title: str,
             fpr_values: List[float] = None,
             log_scale: bool = True,
             save_path: str = None):
    """
    Plot the AUC graph for the predictions from different attacks, the code is adapted from tensorflow_privacy.
    :param pred_list: List of predictions.
    :param name_list: List of names for the attacks.
    :param ground_truth_arr: Ground truth array.
    :param title: Title of the graph.
    :param fpr_values: list of FPR values to plot vertical lines
    :param log_scale: Whether to plot in log scale. Defaults to True.
    :param save_path: Path to save the graph.
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

    if log_scale:
        plt.semilogx()
        plt.semilogy()
    else:
        plt.xscale('linear')
        plt.yscale('linear')

    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.legend(fontsize=8)

    # Draw vertical lines on specified FPR values
    if fpr_values:
        for fpr_value in fpr_values:
            plt.axvline(x=fpr_value, color='r', linestyle='--', linewidth=1)
            plt.text(fpr_value, 0.5, f'FPR={fpr_value:.3f}', color='r', rotation=90)

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
