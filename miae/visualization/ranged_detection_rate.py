import json
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset

from utils.datasets.loader import load_dataset
from visualization.base import BaseVisualization


def _normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst]


def _plot_mia_sensitivity(mia_results, freq, outlier_scores, total_target_models, ax):
    # Initialize arrays to hold frequencies
    freq_not_in_train_but_detected = np.zeros(mia_results.shape[2])
    freq_in_train_and_detected = np.zeros(mia_results.shape[2])

    # Calculate frequencies
    for trial in mia_results:
        mia_predictions, ground_truths = trial
        for i, (mia_prediction, ground_truth) in enumerate(zip(mia_predictions, ground_truths)):
            if not ground_truth and mia_prediction:
                freq_not_in_train_but_detected[i] += 1
            elif ground_truth and mia_prediction:
                freq_in_train_and_detected[i] += 1

    # normalize the data
    for i in range(len(freq_not_in_train_but_detected)):
        freq_not_in_train_but_detected[i] /= total_target_models - freq[i]
    for i in range(len(freq_in_train_and_detected)):
        freq_in_train_and_detected[i] /= freq[i]

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Not in Train, Detected by MIA': freq_not_in_train_but_detected,
        'In Train, Detected by MIA': freq_in_train_and_detected
    })

    # Calculate scores given by y/(x + small number)
    small_number = 1e-10
    x = df['Not in Train, Detected by MIA'].values
    y = df['In Train, Detected by MIA'].values
    return _normalize(y / (x + small_number))

def _preprocess_prediction(lira_stats):
    # Note we need to reverse the score, so <= not >= 0 indicate the true
    return [[[x <= 0 for x in sublist1], sublist2] for sublist1, sublist2 in lira_stats]


def _filter_array(target, target_freq, labels, current_class):
    # Iterate over the first dimension of the larger array
    mask = [label == current_class for label in labels]

    return target[:, :, mask], target_freq[mask]


def _filter_based_on_outliers(target, outlier_scores, threshold):
    outlier_scores = np.array(outlier_scores)
    mask = outlier_scores >= threshold  # This will create a boolean numpy array

    return target[:, :, mask], outlier_scores[mask]


def plot_scores_by_epoch(scores, epochs, metric_score, threshold=0.2, threshold_range=0.2):
    """Visualize the scores for each epoch using colored lines."""
    # Step 1: Normalize the metric_score (assuming normalization means scaling between 0 and 1)
    normalized_metric_score = (metric_score - np.min(metric_score)) / (np.max(metric_score) - np.min(metric_score))

    # Step 2: Find the indices
    selected_indices = np.where((normalized_metric_score >= threshold - threshold_range)
                                & (normalized_metric_score <= threshold))[0]

    # Step 3 & 4: Filter scores based on selected indices and plot for each epoch
    plt.figure(figsize=(15, 50))
    for i, epoch_scores in enumerate(scores):
        epoch_scores = np.array(epoch_scores)
        filtered_scores = epoch_scores[selected_indices]

        plt.subplot(12, 2, i + 1)
        plt.hist(filtered_scores, bins=50, color='skyblue', edgecolor='black')

        plt.yscale('log')
        plt.title(f'Epoch: {epochs[i]}')
        plt.xlabel('Score Value')
        plt.ylabel('Frequency')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


class DetectionRate(BaseVisualization):
    def __init__(self, config):
        super().__init__(config)
        self.num_targets = self.config.get('num_targets', 370)
        self.outlier_method = self.config.get('outlier_method', None)
        self.arch = self.config.get('arch', 'wrn28-2')
        self.epoch = self.config.get('epoch', 140)
        self.interval = self.config.get('interval', 10)
        self.color_metric = self.config.get('color_metric', "prediction_depth")

        self.target_models_dir = self.config.get('outlier_dir', f"../exp/{self.arch}/{self.epoch}/lira_stats_"
                                                                f"{self.num_targets}_target.pkl")
        self.metric_dir = self.config.get('metric_dir', f"../metric_data/{self.color_metric}")

        with open(self.target_models_dir, 'rb') as f:
            self.lira_stats_data = pickle.load(f)

        # Create the freq in train for each data pts
        self.freq = np.sum([item[1] for item in self.lira_stats_data], axis=0)

        dataset = load_dataset(self.config['dataset_name'],
                               self.config['data_path'], None, None, None)
        self.fullset = ConcatDataset([dataset.train_set, dataset.test_set])
        self.fullset_targets = dataset.train_set.targets + dataset.test_set.targets

    def get_plot(self):
        """
        get the score of the metric for the sample with index idx
        :param idx: index of the sample
        """
        # Load the JSON files
        with open(os.path.join(self.metric_dir, 'train.json'), 'r') as train_file:
            train_data = json.load(train_file)

        with open(os.path.join(self.metric_dir, 'test.json'), 'r') as test_file:
            test_data = json.load(test_file)

        # Convert the dictionaries to dataframes
        train_df = pd.DataFrame(list(train_data.items()), columns=['index', 'Score'])
        test_df = pd.DataFrame(list(test_data.items()), columns=['index', 'Score'])

        # Set the 'Index' column as the index
        train_df.set_index('index', inplace=True)
        test_df.set_index('index', inplace=True)

        # Combine the dataframes
        metric_total_df = pd.concat([train_df, test_df])
        metric_total_df.reset_index(drop=True, inplace=True)

        scores = []
        for epoch, in range(10, self.epoch, self.interval):
            with open(f'../exp/{self.arch}/{self.epoch}/lira_stats_{self.num_targets}_target.pkl', 'rb') as f:
                lira_stats_data = pickle.load(f)

            # Create the freq in train for each data pts
            freq = np.sum([item[1] for item in lira_stats_data], axis=0)

            preprocessed_data = _preprocess_prediction(lira_stats_data)
            scores.append(_plot_mia_sensitivity(np.array(preprocessed_data), freq, self.num_targets))

        # Plot binned average scores across epochs
        plot_scores_by_epoch(scores, range(10, self.epoch, self.interval), metric_total_df['Score'].values, threshold=1)

    def __repr__(self):
        return (f"DetectionRate(config={self.config}, num_targets={self.num_targets}, "
                f"outlier_method={self.outlier_method}, arch={self.arch}, epoch={self.epoch}, "
                f"color_metric={self.color_metric}, target_models_dir={self.target_models_dir}, "
                f"metric_dir={self.metric_dir})")
