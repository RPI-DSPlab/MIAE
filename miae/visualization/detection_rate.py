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

    # Plot scatterplot
    sc = ax.scatter(df['Not in Train, Detected by MIA'], df['In Train, Detected by MIA'],
                    s=1, c=outlier_scores, cmap='viridis')
    plt.colorbar(sc)

    # Calculate polynomial fit
    x = df['Not in Train, Detected by MIA'].values
    y = df['In Train, Detected by MIA'].values
    z = np.polyfit(x, y, deg=1)
    p = np.poly1d(z)

    # Plot polynomial fit
    ax.plot(x, p(x), color='red')

    # Add a diagonal line
    diag = np.linspace(start=min(ax.get_xlim()[0], ax.get_ylim()[0]),
                       stop=max(ax.get_xlim()[1], ax.get_ylim()[1]),
                       num=100)
    ax.plot(diag, diag, color='grey', linestyle='dashed')


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


class DetectionRate(BaseVisualization):
    def __init__(self, config):
        super().__init__(config)
        self.num_targets = self.config.get('num_targets', 370)
        self.outlier_method = self.config.get('outlier_method', None)
        self.arch = self.config.get('arch', 'wrn28-2')
        self.epoch = self.config.get('epoch', 140)
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
        # Create subplots
        fig, axs = plt.subplots(2, 5, figsize=(15, 10))
        axs = axs.flatten()

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

        all_scores = []  # Record all the scores

        for i in range(10):
            metric_class_df = metric_total_df[
                metric_total_df.index.isin([idx for idx, label in enumerate(self.fullset_targets) if label == i])]

            scaler = MinMaxScaler()
            score_list = scaler.fit_transform(metric_class_df[['Score']]).flatten()

            filtered_data, preprocessed_freq = _filter_array(np.array(self.lira_stats_data),
                                                             self.freq, self.fullset_targets, i)
            filtered_data, score_list = _filter_based_on_outliers(filtered_data, score_list, 0)
            preprocessed_data = _preprocess_prediction(filtered_data)
            _plot_mia_sensitivity(np.array(preprocessed_data), preprocessed_freq, score_list,
                                  self.num_targets, axs[i])
            axs[i].set_title(f'Class: {i}')

            all_scores.append(score_list)  # Append the score_list to all_scores list

        fig.suptitle('Sensitivity to Membership Inference Attack', size=22)
        plt.subplots_adjust(top=0.9)

        plt.show()

    def __repr__(self):
        return (f"DetectionRate(config={self.config}, num_targets={self.num_targets}, "
                f"outlier_method={self.outlier_method}, arch={self.arch}, epoch={self.epoch}, "
                f"color_metric={self.color_metric}, target_models_dir={self.target_models_dir}, "
                f"metric_dir={self.metric_dir})")
