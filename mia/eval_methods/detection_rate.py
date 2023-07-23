import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset

from sample_metrics.base import ExampleMetric
from utils.datasets.loader import load_dataset


def _normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst]


class DetectionRate(ExampleMetric):
    def __init__(self, config):
        super().__init__(config)
        self.num_targets = self.config.get('num_targets', 370)
        self.outlier_method = self.config.get('outlier_method', None)
        self.outlier_dir = f"../outlier/res/{self.outlier_method}"

        with open(f'../exp/lira_stats_{self.num_targets}_target.pkl', 'rb') as f:
            self.lira_stats_data = pickle.load(f)

        # Create the freq in train for each data pts
        self.freq = np.sum([item[1] for item in self.lira_stats_data], axis=0)

        dataset = load_dataset(self.config['dataset_name'],
                               self.config['data_path'], None, None)
        self.fullset = ConcatDataset([dataset.train_set, dataset.test_set])
        self.fullset_targets = dataset.train_set.targets + dataset.test_set.targets

    def plot_mia_sensitivity(self, mia_results, freq, outlier_scores, total_target_models, ax):
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

    def preprocess_prediction(self, lira_stats):
        # Note we need to reverse the score, so <= not >= 0 indicate the true
        return [[[x <= 0 for x in sublist1], sublist2] for sublist1, sublist2 in lira_stats]

    def filter_array(self, target, target_freq, labels, current_class):
        # Iterate over the first dimension of the larger array
        mask = np.concatenate([np.full(50000, False), [label == current_class for label in labels]])
        # for full set in outlier
        # mask = [label == current_class for label in labels]

        return target[:, :, mask], target_freq[mask]

    def filter_based_on_outliers(self, target, outlier_scores, threshold):
        outlier_scores = np.array(outlier_scores)  # Convert outlier_scores to numpy array
        mask = outlier_scores >= threshold  # This will create a boolean numpy array

        return target[:, :, mask], outlier_scores[mask]

    def get_score(self, idx: int):
        """
        get the score of the metric for the sample with index idx
        :param idx: index of the sample
        """
        # Create subplots
        fig, axs = plt.subplots(2, 5, figsize=(15, 10))
        axs = axs.flatten()

        # Create a plot for each threshold
        outlier_method = "LOF"
        outlier_dir = f"../outlier/res/{outlier_method}"
        for i in range(10):
            outlier_df = pd.read_csv(f"{outlier_dir}/class_{i}_outlier_scores.csv")
            score_list = _normalize(outlier_df['Score'].tolist())
            filtered_data, preprocessed_freq = self.filter_array(np.array(self.lira_stats_data), self.freq,
                                                                 self.fullset_targets, i)
            filtered_data, score_list = self.filter_based_on_outliers(filtered_data, score_list, 0)
            preprocessed_data = self.preprocess_prediction(filtered_data)
            self.plot_mia_sensitivity(np.array(preprocessed_data), preprocessed_freq, score_list,
                                      self.num_targets, axs[i])
            axs[i].set_title(f'Class: {i}')

        fig.suptitle('Sensitivity to Membership Inference Attack', size=22)
        plt.subplots_adjust(top=0.9)

        plt.show()

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def load(self, path):
        """
        load the metric from the path
        :param path: path to load the metric
        """
        pass
