import pickle

import numpy as np
from matplotlib import pyplot as plt


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

        # print the length of the true positive samples on the plot
        plt.text(0.5, 0.5, f"TP samples: {len(tp)}", fontsize=12, transform=plt.gca().transAxes)

        # Save the plot
        plt.savefig(save_path, dpi=300)

    def normalize(self):
        """
        Normalize the sample hardness scores to the range [0, 1].
        This is a generator function.
        """
        new_score_arr = (self.score_arr - self.min_score) / (self.max_score - self.min_score)
        return SampleHardness(new_score_arr, self.name)


def load_sample_hardness(file_path: str) -> np.ndarray:
    """
    Load example hardness scores from a file.

    :param file_path: path to the file containing the example hardness scores
    :return: example hardness scores as a numpy array
    """
    if file_path.endswith(".npy") or file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            example_hardness = pickle.load(f)
        return example_hardness
    else:
        raise ValueError("File format not supported.")
