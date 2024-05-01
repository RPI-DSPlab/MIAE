from typing import List

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_curve


def pred_normalization(pred: np.ndarray) -> np.ndarray:
    """
    Normalize the predictions to [0, 1].

    :param pred: predictions as a numpy array
    :return: normalized predictions
    """
    if pred.dtype == bool:
        pred = pred.astype(int)
    return (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-6)


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
        pred_norm = pred_normalization(self.pred_arr)
        labels = (pred_norm > threshold).astype(int)
        return labels

    def accuracy(self, threshold=0.5) -> float:
        """
        Calculate the accuracy of the predictions.

        :param self: Predictions object
        :param threshold: threshold for converting predictions to binary labels
        :return: accuracy of the predictions
        """
        return np.mean(self.predictions_to_labels(threshold) == self.ground_truth_arr)

    def balanced_attack_accuracy(self) -> float:
        """
        Calculate the balanced attack accuracy for a single attack.

        :param pred: Predictions object
        :return: balanced attack accuracy of the predictions
        """
        return balanced_accuracy_score(self.ground_truth_arr, self.predictions_to_labels())

    def compute_fpr(self):
        """
        Compute the false positive rate (FPR) of the predictions.
        """
        pred_tensor = torch.tensor(self.pred_arr)
        ground_truth_tensor = torch.tensor(self.ground_truth_arr)
        false_positive = torch.logical_and(pred_tensor == 1, ground_truth_tensor == 0).sum().item()
        true_negative = torch.logical_and(pred_tensor == 0, ground_truth_tensor == 0).sum().item()
        total_negative = true_negative + false_positive
        FPR = false_positive / total_negative if total_negative > 0 else 0
        return FPR

    def compute_tpr(self):
        """
        Compute the true positive rate (TPR) of the predictions.
        """
        pred_tensor = torch.tensor(self.pred_arr)
        ground_truth_tensor = torch.tensor(self.ground_truth_arr)
        true_positive = torch.logical_and(pred_tensor == 1, ground_truth_tensor == 1).sum().item()
        false_negative = torch.logical_and(pred_tensor == 0, ground_truth_tensor == 1).sum().item()
        total_positive = true_positive + false_negative
        TPR = true_positive / total_positive if total_positive > 0 else 0
        return TPR

    def adjust_fpr(self, target_fpr):
        """
        Adjust the predictions to achieve a target FPR using ROC curve.
        :param target_fpr: target FPR
        :return: adjusted predictions as a numpy array
        """
        fpr, tpr, thresholds = roc_curve(self.ground_truth_arr, self.pred_arr)

        # Find the threshold closest to the target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        threshold = thresholds[idx]

        # Adjust predictions based on the selected threshold
        adjusted_pred_arr = (self.pred_arr >= threshold).astype(int)

        return adjusted_pred_arr

    def get_tp(self) -> np.ndarray:
        """
        Get the indices of the true positive samples.
        """
        return np.where((self.predictions_to_labels() == 1) & (self.ground_truth_arr == 1))[0]

    def tpr_at_fpr(self, fpr: float) -> float:
        """
        Compute TPR at a specified FPR.

        :param fpr: FPR value
        :return: TPR value
        """
        adjusted_pred = self.adjust_fpr(fpr)
        pred_tensor = torch.tensor(adjusted_pred)
        ground_truth_tensor = torch.tensor(self.ground_truth_arr)
        true_positive = torch.logical_and(pred_tensor == 1, ground_truth_tensor == 1).sum().item()
        false_negative = torch.logical_and(pred_tensor == 0, ground_truth_tensor == 1).sum().item()
        total_positive = true_positive + false_negative
        tpr = true_positive / total_positive if total_positive > 0 else 0

        return tpr

    def __len__(self):
        """
        return the length of the prediction array
        """

        return len(self.pred_arr)


def _common_tp(preds: List[Predictions], fpr=None, threshold=0.5, set_op="intersection"):
    """
    Find the union/intersection true positive samples among the predictions
    Note that this is used for both different attacks or same attack with different seeds.

    :param preds: list of Predictions
    :param fpr: FPR values for adjusting the predictions
    :param threshold: threshold for converting predictions to binary labels (only used when not using fpr)

    :return: common true positive samples
    """
    if fpr is None:
        TP = [np.where((pred.predictions_to_labels(threshold) == 1) & (pred.ground_truth_arr == 1))[0] for pred in
              preds]
    else:
        adjusted_preds = [pred.adjust_fpr(fpr) for pred in preds]
        TP = [np.where((adjusted_preds[i] == 1) & (preds[i].ground_truth_arr == 1))[0] for i in range(len(preds))]
    common_TP = set(TP[0])
    if len(TP) < 2:
        return common_TP
    for i in range(1, len(TP)):
        if set_op == "union":
            common_TP = common_TP.union(set(TP[i]))
        elif set_op == "intersection":
            common_TP = common_TP.intersection(set(TP[i]))
    return common_TP


def union_tp(preds: List[Predictions], fpr=None):
    """
    Find the union true positive samples among the predictions, it's a wrapper for common_tp
    """
    return _common_tp(preds, fpr, set_op="union")


def intersection_tp(preds: List[Predictions], fpr=None):
    """
    Finds the intersection true positive samples among the predictions, it's a wrapper for common_tp
    """
    return _common_tp(preds, fpr, set_op="intersection")


def _common_pred(preds: List[Predictions], fpr=None, threshold=0.5, set_op="intersection"):
    """
    Find the union/intersection of prediction = 1 among the predictions
    Note that this is used for both different attacks or same attack with different seeds.

    :param preds: list of Predictions
    :param fpr: FPR values for adjusting the predictions
    :param threshold: threshold for converting predictions to binary labels (only used when not using fpr)
    :param set_op: set operation for the common predictions: [union, intersection]
    """
    if fpr is None:
        pred = [np.where(pred.predictions_to_labels(threshold) == 1)[0] for pred in preds]
    else:
        adjusted_preds = [pred.adjust_fpr(fpr) for pred in preds]
        pred = [np.where(adjusted_preds[i] == 1)[0] for i in range(len(preds))]

    common_pred = set(pred[0])
    if len(pred) < 2:
        return common_pred

    for i in range(1, len(pred)):
        if set_op == "union":
            common_pred = common_pred.union(set(pred[i]))
        elif set_op == "intersection":
            common_pred = common_pred.intersection(set(pred[i]))
    return common_pred


def union_pred(preds: List[Predictions], fpr=None):
    """
    Find the union of prediction = 1 among the predictions, it's a wrapper for common_pred
    """
    return _common_pred(preds, fpr, set_op="union")


def intersection_pred(preds: List[Predictions], fpr=None):
    """
    Find the intersection of prediction = 1 among the predictions, it's a wrapper for common_pred
    """
    return _common_pred(preds, fpr, set_op="intersection")


def multi_seed_ensemble(pred_list: List[Predictions], method, threshold: float = None,
                        fpr: float = None) -> Predictions:
    """
    Ensemble the predictions from different seeds of the same attack.

    :param pred_list: list of Predictions
    :param method: method for ensemble the predictions: [HC, HP, avg]
    :param threshold: threshold for ensemble the predictions
    :param fpr: FPR values for adjusting the predictions
    :return: ensemble prediction
    """
    if threshold is not None and fpr is not None:
        raise ValueError("Both threshold and FPR values are provided, only one should be provided.")
    if len(pred_list) < 2:
        return pred_list[0]

    ensemble_pred = np.zeros_like(pred_list[0].pred_arr)
    if method == "HC":  # High Coverage
        agg_tp = list(_common_pred(pred_list, set_op="union", threshold=threshold))
        if len(agg_tp) > 0:
            ensemble_pred[agg_tp] = 1
        else:
            print("No common true positive samples found for the ensemble (HC).")
    elif method == "HP":  # High Precision
        agg_tp = list(_common_pred(pred_list, set_op="intersection", threshold=threshold))
        if len(agg_tp) > 0:
            ensemble_pred[agg_tp] = 1
        else:
            print("No common true positive samples found for the ensemble (HP).")
    elif method == "avg":  # averaging
        ensemble_pred = np.mean([pred.pred_arr for pred in pred_list], axis=0)
    else:
        raise ValueError("Invalid method for ensemble the predictions.")

    return Predictions(ensemble_pred, pred_list[0].ground_truth_arr, f"ensemble_{method}")


def pred_tp_intersection(pred_list: List[Predictions]) -> Predictions:
    """
    Get the common true positive predictions across different seeds of a single attack
    this is used for the Venn diagram

    :param pred_list: List of Predictions objects for the same attack but different seeds
    :return: Predictions object containing only common true positives
    """
    if len(pred_list) < 2:
        raise ValueError("At least 2 predictions are required for comparison.")

    # get the common true positive predictions using logical and
    common_tp_or = pred_list[0].predictions_to_labels()
    common_tp_and = pred_list[0].predictions_to_labels()
    for i in range(1, len(pred_list)):
        common_tp_or = np.logical_or(common_tp_or, pred_list[i].predictions_to_labels())  # union
        common_tp_and = np.logical_and(common_tp_and, pred_list[i].predictions_to_labels())  # intersection

    # create a new Predictions object for the common true positive predictions
    ground_truth_arr = pred_list[0].ground_truth_arr
    name = pred_list[0].name.split('_')[0]
    pred_or = Predictions(common_tp_or, ground_truth_arr, name)
    pred_and = Predictions(common_tp_and, ground_truth_arr, name)
    return pred_or, pred_and


def averaging_predictions(pred_list: List[Predictions]) -> np.ndarray:
    """
    Average the predictions from different attacks.

    :param pred_list: list of Predictions
    :return: averaged prediction
    """
    pred_list = [pred.pred_arr for pred in pred_list]
    return np.mean(pred_list, axis=0)


def majority_voting(pred_list: List[Predictions], threshold=None | float, ) -> np.ndarray:
    """
    Majority voting for the predictions from different attacks.

    :param pred_list: list of Predictions
    :return: majority voted prediction
    """
    # convert predictions to binary labels
    labels_list = [pred.predictions_to_labels(threshold=threshold) for pred in pred_list]

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
