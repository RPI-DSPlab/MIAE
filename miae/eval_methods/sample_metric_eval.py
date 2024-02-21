import numpy as np
import matplotlib.pyplot as plt
import math


def read_metric(path: str):
    """
    read the metric(dict saved to json) from path
    :param path: str, path to the metric file
    :return: dict, key: sample index, value: score
    """
    import json
    with open(path, 'r') as f:
        metric = json.load(f)
    return metric


class SampleMetric:
    def __init__(self, train_score_dict: dict, test_score_dict: dict, train_loader, test_loader):
        """
        initialize the sample metric
        :param train_score_dict: dict, key: sample index, value: score
        :param test_score_dict: dict, key: sample index, value: score
        :param train_loader: train data loader
        :param test_loader: test data loader
        """
        self.train_score_dict = train_score_dict
        self.test_score_dict = test_score_dict
        self.train_loader = train_loader
        self.test_loader = test_loader

    def print_score_len(self):
        print(f"train score len: {len(self.train_score_dict)}, test score len: {len(self.test_score_dict)}")

    def plot_score_distribution(self, bucket_option: str | int = None):
        """
        plot the score distribution for train and test
        :param bucket_option: str | int, if None, each bucket corresponds to a score, if None, use auto bucket option in numpy
        :return: plt object with 2 subplots for train and test
        """

        train_scores = self.train_score_dict.values()
        if bucket_option is None:  # auto bucket means each 2 buckets corresponds to a score
            num_bin = math.ceil(max(train_scores)) * 2
        else:
            num_bin = 'auto'

        plt.subplot(1, 2, 1)
        plt.hist(train_scores, bins=num_bin, alpha=0.5, label='train')
        plt.legend(loc='upper right')

        test_scores = self.test_score_dict.values()
        plt.subplot(1, 2, 2)
        plt.hist(test_scores, bins=num_bin, alpha=0.5, label='test')
        plt.legend(loc='upper right')

        return plt

    def plot_2_score_distribution(self, other_sm, bucket_option: str | int = None):
        """
        plot the score distribution
        :param bucket_option: str | int, if None, each bucket corresponds to a score, if None, use auto bucket option in numpy
        :return: plt object with 2 subplots for train and test, and correlation coefficient for train and test
        """

        train_scores = self.train_score_dict.values()
        other_train_scores = other_sm.train_score_dict.values()

        if bucket_option is None:  # auto bucket means each 2 buckets corresponds to a score
            num_bin = math.ceil(max(train_scores)) * 2
        else:
            num_bin = 'auto'

        plt.subplot(1, 2, 1)
        plt.hist(train_scores, bins=num_bin, alpha=0.5, label='train')
        plt.hist(other_train_scores, bins=num_bin, alpha=0.5, label='other train')
        plt.legend(loc='upper right')

        test_scores = self.test_score_dict.values()
        other_test_scores = other_sm.test_score_dict.values()
        plt.subplot(1, 2, 2)
        plt.hist(test_scores, bins=num_bin, alpha=0.5, label='test')
        plt.hist(other_test_scores, bins=num_bin, alpha=0.5, label='other test')
        plt.legend(loc='upper right')

        # calculate correlation coefficient
        train_corr = np.corrcoef(train_scores, other_train_scores)[0, 1]
        test_corr = np.corrcoef(test_scores, other_test_scores)[0, 1]

        return plt, train_corr, test_corr

