# This code implements "Systematic evaluation of privacy risks of machine learning models", USENIX Security 2021
# The code is based on the code from
# https://github.com/inspire-group/membership-inference-evaluation

import torch
from matplotlib import pyplot as plt

from attack_classifier import *
from attacks.base import MiAttack, AuxiliaryInfo, ModelAccessType, ModelAccess


class ThreshEntropyMiaAuxiliaryInfo(AuxiliaryInfo):
    """
    Base class for all auxiliary information.
    """

    def __init__(self, config):
        # Array-based mock confidences for demonstration purposes
        self.s_tr_conf = config.get("s_tr_conf", np.array([0.75, 0.65]))
        self.s_te_conf = config.get("s_te_conf", np.array([0.5, 0.55]))
        self.num_classes = config.get("num_classes", 10)


def compute_distributions(tr_values, te_values, tr_labels, te_labels, num_bins=30, log_bins=True,
                          plot_name=None):
    """
    Computes and plots histograms for training and test values.
    """
    num_classes = len(np.unique(np.concatenate((tr_labels, te_labels))))
    tr_distrs = []
    te_distrs = []
    all_bins = []

    for c in range(num_classes):
        tr_data_c = tr_values[tr_labels == c]
        te_data_c = te_values[te_labels == c]

        data_range = (min(np.min(tr_data_c), np.min(te_data_c)), max(np.max(tr_data_c), np.max(te_data_c)))

        if log_bins:
            bins = np.logspace(np.log10(data_range[0]), np.log10(data_range[1]), num_bins)
            plt.xscale('log')
        else:
            bins = np.linspace(data_range[0], data_range[1], num_bins)

        all_bins.append(bins)

        tr_hist, _ = np.histogram(tr_data_c, bins=bins, density=True)
        te_hist, _ = np.histogram(te_data_c, bins=bins, density=True)

        tr_distrs.append(tr_hist)
        te_distrs.append(te_hist)

        plt.hist(tr_data_c, bins=bins, color='b', alpha=0.5, weights=[1. / len(tr_data_c)] * len(tr_data_c),
                 label='Train')
        plt.hist(te_data_c, bins=bins, color='r', alpha=0.5, weights=[1. / len(te_data_c)] * len(te_data_c),
                 label='Test')
        plt.legend()
        plt.show()

    if plot_name:
        plt.savefig(plot_name)
    else:
        plt.savefig('./tmp.png')

    return tr_distrs, te_distrs, all_bins


class ThreshEntropyMiaModelAccess(ModelAccess):
    """
    Base class for all types of model access.
    """

    def __init__(self, model, num_classes):
        super().__init__(model, ModelAccessType.BLACK_BOX)
        self.num_classes = num_classes

    def get_signal(self, data):
        # Get model predictions
        predictions = self.model(data)

        # Extract the confidence values
        confidences = np.max(predictions, axis=1)

        # Compute entropy and modified entropy for each prediction
        entr = -np.sum(predictions * np.log(predictions + 1e-30), axis=1)
        m_entr = -np.sum(predictions * np.log(1 - predictions + 1e-30), axis=1)

        return confidences, entr, m_entr


class ThreshEntropyMia(MiAttack):
    """
    Base class for all attacks.
    """

    # define initialization with specifying the model access and the auxiliary information
    def __init__(self, target_model_access: ModelAccess, auxiliary_info: ThreshEntropyMiaAuxiliaryInfo, target_data=None):
        """
        Initialize the attack with model access and auxiliary information.
        :param target_model_access:
        :param auxiliary_info:
        :param target_data: if target_data is not None, the attack could be data dependent. The target data is used to
        develop the attack model or classifier.
        """
        super().__init__(target_model_access, auxiliary_info, target_data)
        self.target_model_access = target_model_access
        self.num_classes = auxiliary_info.num_classes
        self.thresholds = None

    def _thre_setting(self, tr_values, te_values):
        # Logic for determining the threshold (from membership_inference_attacks.py)
        optimal_thre = None
        optimal_acc = 0
        for thre in np.concatenate((tr_values, te_values)):
            acc = np.mean(tr_values > thre) + np.mean(te_values <= thre)
            if acc > optimal_acc:
                optimal_acc = acc
                optimal_thre = thre
        return optimal_thre

    def prepare(self, auxiliary_info: AuxiliaryInfo):
        # Logic to setup the attack, including determining optimal thresholds
        s_tr_conf, s_te_conf = auxiliary_info
        self.thresholds = [self._thre_setting(s_tr_conf[i], s_te_conf[i]) for i in range(self.num_classes)]

    def build_attack_classifier(self, classifer_config: dict):
        # In our case, the "attack classifier" is essentially the thresholds
        return self.thresholds

    def infer(self, target_data):
        # Use the thresholds to infer membership for target_data
        confidences, _, _ = self.target_model_access.get_signal(target_data)
        inferred_membership = [1 if confidences[i] > self.thresholds[i] else 0 for i in range(self.num_classes)]
        return inferred_membership


