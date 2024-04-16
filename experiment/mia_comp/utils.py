"""
This file defines classes/functions for comparing the MIA's predictions down to sample level.
"""
import os
import pickle
import sys

from MIAE.miae.eval_methods.prediction import Predictions

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import models
from typing import Optional, Callable, Union
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from typing import List, Tuple

fprs_sampling = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


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


def extract_features(model, dataset, target_layer=None):
    """
    Extract features from a given model for each image in the dataset.

    :param model: the pre-trained model
    :param dataset: the dataset
    :param target_layer: optional, the layer from which to extract features
    :return: extracted features
    """
    # Set model to evaluation mode
    model.eval()

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract features from the chosen layer
    features = []
    print(f"the length of the dataset is {len(dataset)}")
    with torch.no_grad():
        for i in range(len(dataset)):
            if i % 1000 == 0:
                print(f"processing the {i}th image")
            img, _ = dataset[i]
            # Apply transformation
            img_transformed = transform(img)
            img_transformed = img_transformed.unsqueeze(0)
            if target_layer is not None:
                feature = model.get_features(img_transformed, target_layer)
            else:
                feature = model.get_features(img_transformed)
            feature = feature.view(feature.size(0), -1)
            features.append(feature.squeeze().numpy())
    return np.array(features)


def plot_t_sne(pred_list: List[Predictions], dataset: ConcatDataset, model_name: str, title: str, save_path: str,
               perplexity: int = 30):
    """
    Plot the t-SNE graph for the predictions from different attacks.

    :param pred_list: list of Predictions from different attacks
    :param dataset: the dataset
    :param model_name: the name of our model
    :param title: title of the graph
    :param save_path: path to save the graph
    :param perplexity: perplexity for t-SNE (default: 30)
    """
    # check if the number of attacks is less than 2
    if len(pred_list) < 2:
        raise ValueError("At least 2 attacks are required for comparison.")

    # get model
    print(f"start loading model {model_name}")
    try:
        model = models.get_model(model_name, 10, 32)
    except ValueError:
        raise ValueError("Invalid model name.")
    print(f"finish loading model {model_name}")

    # Extract features from the chosen model
    print(f"start extracting features from the model {model_name}")
    high_dim_data = extract_features(model, dataset)
    print(f"finish extracting features from the model {model_name}")

    # Perform t-SNE on the extracted features
    print(f"apply t-SNE on the extracted features")
    tsne = TSNE(n_components=2, perplexity=perplexity)
    low_dim_data = tsne.fit_transform(high_dim_data)
    print(f"get low dimensional data from t-SNE, its shape is {low_dim_data.shape}")

    # Plot the t-SNE graph
    attack_names = [pred.name for pred in pred_list]
    plt.figure(figsize=(10, 10), dpi=300)
    for i, pred in enumerate(pred_list):
        indices = np.where((pred.predictions_to_labels() == 1) & (pred.ground_truth_arr == 1))[0]
        color = plt.cm.tab10(i)
        plt.scatter(low_dim_data[indices, 0], low_dim_data[indices, 1], label=attack_names[i], s=1, color=color)

    # Set limits for the plot based on the range of low-dimensional data
    min_x, min_y = np.min(low_dim_data, axis=0)
    max_x, max_y = np.max(low_dim_data, axis=0)
    margin = 5
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


