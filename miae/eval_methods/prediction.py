import os
from typing import List, Union, Tuple
from typing import List, Union, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
import csv

from miae.utils.roc_auc import fig_fpr_tpr

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
    """
    Predictions class stores the predictions and ground truth for a single attack instance.
    It could be either hard label predictions or soft label predictions.
    If it's hard label, the predictions are considered as "membership predictions".
    If it's soft label, the predictions are considered as "membership scores".
    """
    def __init__(self, pred_arr: np.ndarray, ground_truth_arr: np.ndarray, name: str):
        """
        Initialize the Predictions object.

        :param pred_arr: predictions as a numpy array
        :param ground_truth_arr: ground truth as a numpy array
        :param name: name of the attack
        """
        if type(pred_arr) != np.ndarray or type(ground_truth_arr) != np.ndarray:
            raise ValueError("The predictions and ground truth should be numpy arrays.")
        self.pred_arr = pred_arr
        self.ground_truth_arr = ground_truth_arr
        self.name = name


    def update_name(self, new_name):
        """
        Update the name of the Predictions object.
        :param new_name: new name of the Predictions object
        """
        self.name = new_name

    def is_hard(self):
        """
        return true if the predictions are hard labels.
        Hard label predictions are considered as "membership predictions", and
        Soft label predictions are considered as "membership scores".
        """
        for i in self.pred_arr:
            if i not in [0.0, 1.0, 0, 1]:
                return False
        return True

    def predictions_to_labels(self, threshold: float = 0.5, normalization=False) -> np.ndarray:
        """
        Convert predictions to binary labels.

        :param self: Predictions object
        :param threshold: threshold for converting predictions to binary labels
        :return: binary labels as a numpy array
        """
        pred = pred_normalization(self.pred_arr) if normalization else self.pred_arr
        labels = (pred > threshold).astype(int)
        return labels

    def accuracy(self, threshold=0.5) -> float:
        """
        Calculate the accuracy of the predictions.

        :param self: Predictions object
        :param threshold: threshold for converting predictions to binary labels
        :return: accuracy of the predictions
        """
        if not self.is_hard():
            return np.mean(self.predictions_to_labels(threshold) == self.ground_truth_arr)
        else:
            return np.mean(self.pred_arr == self.ground_truth_arr)

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
        if not self.is_hard():
            raise ValueError("The predictions are not hard labels.")
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
        if not self.is_hard():
            raise ValueError("The predictions are not hard labels.")
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

        if self.is_hard():
            raise ValueError("The predictions are already hard label (0, 1), fpr is already determined.")

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

    def tnr_at_fpr(self, fpr: float) -> float:
        """
        Compute TNR at a specified FPR.
        :param fpr: FPR value
        :return: TNR value
        """
        adjusted_pred = self.adjust_fpr(fpr)
        pred_tensor = torch.tensor(adjusted_pred)
        ground_truth_tensor = torch.tensor(self.ground_truth_arr)
        true_negative = torch.logical_and(pred_tensor == 0, ground_truth_tensor == 0).sum().item()
        false_positive = torch.logical_and(pred_tensor == 1, ground_truth_tensor == 0).sum().item()
        total_negative = true_negative + false_positive
        tnr = true_negative / total_negative if total_negative > 0 else 0
        return tnr

    def __len__(self):
        """
        return the length of the prediction array
        """

        return len(self.pred_arr)


    def save_pred(self, file_name: str, path: str):
        """
        Save the prediction to a file.
        """
        np.save(os.path.join(path, file_name), self.pred_arr)


    def roc_curve(self):
        """
        Compute the ROC curve of the predictions with sklearn roc_curve.
        """
        fpr, tpr, threshold = roc_curve(self.ground_truth_arr, self.pred_arr)
        return fpr, tpr, threshold
    
    def confusion_matrix_precision(self):
        """
        Compute the precision under the confusion matrix.Precision = TP / (TP + FP)
        """
        if not self.is_hard():
            raise ValueError("Precision metric requires hard label predictions")

        TP = np.sum((self.predictions_to_labels() == 1) & (self.ground_truth_arr == 1))
        FP = np.sum((self.predictions_to_labels() == 1) & (self.ground_truth_arr == 0))
        if (TP + FP) == 0:
            print(f"prediction {self.name} has no positive samples")
        return TP / (TP + FP)
    
    def jaccard_similarity(self, other_pred, **kwargs) -> float:
        """
        Compute the Jaccard similarity between two predictions.
        Jaccard similarity = Intersection / Union
        """
        if not self.is_hard():
            raise ValueError("Jaccard similarity metric requires hard label predictions")
        if not other_pred.is_hard():
            raise ValueError("Jaccard similarity metric requires hard label predictions")

        if "canary_indices" in kwargs:
            # focus on the canary indices of the predictions
            canary_indices = kwargs["canary_indices"]
            self_pred_arr = self.predictions_to_labels()[canary_indices]
            other_pred_arr = other_pred.predictions_to_labels()[canary_indices]
        else:
            self_pred_arr = self.predictions_to_labels()
            other_pred_arr = other_pred.predictions_to_labels()

        intersection = np.sum((self_pred_arr == 1) & (other_pred_arr == 1))
        union = np.sum((self_pred_arr == 1) | (other_pred_arr == 1))
        return intersection / union


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


def find_common_tp_pred(pred_list: List[Predictions], fpr) -> Predictions:
    """
    Get the common true positive predictions across different seeds of a single attack
    this is used for the Venn diagram

    :param pred_list: List of Predictions objects for the same attack but different seeds
    :param fpr: FPR value for adjusting the predictions
    :return: Predictions object containing only common true positives
    """
    if len(pred_list) < 2:
        raise ValueError("At least 2 predictions are required for comparison.")

    common_tp_union_indices = union_tp(pred_list, fpr=fpr)
    common_tp_union = np.zeros_like(pred_list[0].pred_arr)
    common_tp_union[list(common_tp_union_indices)] = 1

    common_tp_intersection_indices = intersection_tp(pred_list, fpr=fpr)
    common_tp_intersection = np.zeros_like(pred_list[0].pred_arr)
    common_tp_intersection[list(common_tp_intersection_indices)] = 1

    name = pred_list[0].name.rsplit('_', 1)[0]
    pred_union = Predictions(common_tp_union, pred_list[0].ground_truth_arr, name + "_Coverage")
    pred_intersection = Predictions(common_tp_intersection, pred_list[0].ground_truth_arr, name + "_Stability")

    return pred_union, pred_intersection


def _common_tn(preds: List[Predictions], fpr=None, threshold=0.5, set_op="intersection"):
    """
    Find the union/intersection true negative samples among the predictions.
    Note that this is used for both different attacks or same attack with different seeds.

    :param preds: list of Predictions
    :param fpr: FPR values for adjusting the predictions
    :param threshold: threshold for converting predictions to binary labels (only used when not using fpr)

    :return: common true negative samples
    """
    if fpr is None:
        TN = [np.where((pred.predictions_to_labels(threshold) == 0) & (pred.ground_truth_arr == 0))[0] for pred in preds]
    else:
        adjusted_preds = [pred.adjust_fpr(fpr) for pred in preds]
        TN = [np.where((adjusted_preds[i] == 0) & (preds[i].ground_truth_arr == 0))[0] for i in range(len(preds))]
    common_TN = set(TN[0])
    if len(TN) < 2:
        return common_TN
    for i in range(1, len(TN)):
        if set_op == "union":
            common_TN = common_TN.union(set(TN[i]))
        elif set_op == "intersection":
            common_TN = common_TN.intersection(set(TN[i]))
    return common_TN


def union_tn(preds: List[Predictions], fpr=None):
    """
    Find the union true negative samples among the predictions, it's a wrapper for common_tn
    """
    return _common_tn(preds, fpr, set_op="union")


def intersection_tn(preds: List[Predictions], fpr=None):
    """
    Finds the intersection true negative samples among the predictions, it's a wrapper for common_tn
    """
    return _common_tn(preds, fpr, set_op="intersection")

def find_common_tn_pred(pred_list: List[Predictions], fpr) -> Predictions:
    """
    Get the common true negative predictions across different seeds of a single attack
    this is used for the Venn diagram

    :param pred_list: List of Predictions objects for the same attack but different seeds
    :param fpr: FPR value for adjusting the predictions
    :return: Predictions object containing only common true negatives
    """
    if len(pred_list) < 2:
        raise ValueError("At least 2 predictions are required for comparison.")

    common_tn_union_indices = union_tn(pred_list, fpr=fpr)
    common_tn_union = np.zeros_like(pred_list[0].pred_arr)
    common_tn_union[list(common_tn_union_indices)] = 1

    common_tn_intersection_indices = intersection_tn(pred_list, fpr=fpr)
    common_tn_intersection = np.zeros_like(pred_list[0].pred_arr)
    common_tn_intersection[list(common_tn_intersection_indices)] = 1

    name = pred_list[0].name.rsplit('_', 1)[0]
    pred_union = Predictions(common_tn_union, pred_list[0].ground_truth_arr, name + "_tn_Coverage")
    pred_intersection = Predictions(common_tn_intersection, pred_list[0].ground_truth_arr, name + "_tn_Stability")

    return pred_union, pred_intersection


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
    pred_name_ensemble = pred_list[0].name.split('_')[0] + f" ensemble_{method}"
    return Predictions(ensemble_pred, pred_list[0].ground_truth_arr, pred_name_ensemble)

def averaging_predictions(pred_list: List[Predictions]) -> np.ndarray:
    """
    Average the predictions from different attacks.

    :param pred_list: list of Predictions
    :return: averaged prediction
    """
    pred_list = [pred.pred_arr for pred in pred_list]
    return np.mean(pred_list, axis=0)


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


def plot_auc(pred_list: List[List[Predictions]] | List[Predictions],
             name_list: List[str],
             title: str,
             fpr_values: List[float] = None,
             save_path: str = None,
             acc_save: str = "False"
             ):
    """
    Plot the AUC graph for the predictions from different attacks with FPR sampling: take the hard label predictions from
    different FPRs and plot the ROC curve.

    :param pred_list: List of lists predictions: [pred1, pred2, ...], where pred1 = [pred1_fpr1, pred1_fpr2, ...]
                        or List of Predictions. (depends on the prediction type)
    :param name_list: List of names for the attacks.
    :param title: Title of the graph.
    :param fpr_values: list of FPR values to plot vertical lines
    :param save_path: Path to save the graph, including the file name.
    :param acc_save: Save the accuracy and AUC values to a file. If False, print instead of saving.
    """

    # get the ground_truth_arr
    if isinstance(pred_list[0], list):
        ground_truth_arr = pred_list[0][0].ground_truth_arr
    elif isinstance(pred_list[0], Predictions):
        ground_truth_arr = pred_list[0].ground_truth_arr
    else:
        raise ValueError("Invalid prediction type.")


    def do_plot_hard(predictions: List[Predictions],
                    legend: str = '',
                    acc_dict: dict = None,
                     **plot_kwargs: Union[int, str, float]) -> Tuple[float, float]:
        """
        Generate the ROC curves for hard label predictions.
        """
        fpr_tpr = []
        for pred in predictions:
            fpr_i = pred.compute_fpr()
            tpr_i = pred.compute_tpr()
            fpr_tpr.append((fpr_i, tpr_i))

        fpr_tpr.sort()
        fpr, tpr = zip(*fpr_tpr)  # unpack the list of tuples
        fpr, tpr = np.array(fpr), np.array(tpr)


        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        auc_score = auc(fpr, tpr)

        low = tpr[np.where(fpr < .001)[0][-1]] if np.any(fpr < .001) else 0


        print(f'Attack: {legend.strip():<20} AUC: {auc_score:<8.4f} max Accuracy: {acc:<8.4f} TPR@0.1%FPR: {low:<8.4f}')

        if acc_dict is not None:
            acc_dict["attack"] = legend.strip()
            acc_dict["AUC"] = auc_score
            acc_dict["max Accuracy"] = acc
            acc_dict["TPR@0.1%FPR"] = low

        metric_text = f'auc={auc_score:.3f}'

        plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)

        return acc, auc_score

    def do_plot_soft(prediction: Predictions,
                     answers: np.ndarray,
                     legend: str = '',
                     acc_list: list = None,
                     **plot_kwargs: Union[int, str, float],) -> Tuple[float, float]:
        """
        Generate the ROC curves for soft label predictions.

        Args:
            prediction (np.ndarray): The predicted scores.
            answers (np.ndarray): The ground truth labels.
            legend (str, optional): Legend for the plot. Defaults to ''.
            acc_list (list, optional): List to store the accuracy and AUC values as dictories. Defaults to None.

        Returns:
            Tuple[float, float]: Accuracy and Area Under the Curve (AUC).
        """
        pred_as_arr = prediction.pred_arr
        def sweep(score: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
            """
             Compute roc curve, auc, and accuracy.

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

        fpr, tpr, auc_score, acc = sweep(np.array(pred_as_arr), np.array(answers, dtype=bool))

        low = tpr[np.where(fpr < .001)[0][-1]] if np.any(fpr < .001) else 0

        print(f'Attack: {legend.strip():<20} AUC: {auc_score:<8.4f} max Accuracy: {acc:<8.4f} TPR@0.1%FPR: {low:<8.4f}')

        if acc_list is not None:
            acc_dict = dict()
            acc_dict["attack"] = legend.strip()
            acc_dict["AUC"] = auc_score
            acc_dict["max Accuracy"] = acc
            acc_dict["TPR@0.1%FPR"] = low
            acc_list.append(acc_dict)

        metric_text = f'auc={auc_score:.3f}'

        plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)

        return acc, auc_score

    plt.figure(figsize=(6, 5))

    membership_list = [ground_truth_arr for _ in range(len(name_list))]
    acc_list = [] if acc_save else None
    for prediction, answer, legend in zip(pred_list, membership_list, name_list):
        if isinstance(prediction, Predictions):
            do_plot_soft(prediction, answer, f"{legend}\n", acc_list)
        elif isinstance(prediction[0], Predictions):
            # there are multiple FPR values
            do_plot_hard(prediction, f"{legend}\n", acc_list)
        else:
            raise ValueError("Invalid prediction type.")

    plt.semilogx()
    plt.semilogy()

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
        format = save_path.split('.')[-1]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.show()
        plt.savefig(save_path, format=format)

        if acc_list is not None:
            # save the accuracy and AUC values to a file
            format = save_path.split('.')[-1]
            with open(save_path.replace(format, "csv"), "w") as f:
                writer = csv.DictWriter(f, fieldnames=acc_list[0].keys())
                writer.writeheader()
                writer.writerows(acc_list)

    return


def get_fpr_tpr_hard_label(pred: np.array, gt: np.array) -> Tuple[float, float]:
    """
    Compute the true positive rate (TPR) and false positive rate (FPR) for hard label predictions.

    :param pred: predicted labels as a numpy array
    :param gt: ground truth labels as a numpy array
    :return: FPR and TPR
    """
    pred_tensor = torch.tensor(pred)
    ground_truth_tensor = torch.tensor(gt)
    true_positive = torch.logical_and(pred_tensor == 1, ground_truth_tensor == 1).sum().item()
    false_positive = torch.logical_and(pred_tensor == 1, ground_truth_tensor == 0).sum().item()
    false_negative = torch.logical_and(pred_tensor == 0, ground_truth_tensor == 1).sum().item()
    true_negative = torch.logical_and(pred_tensor == 0, ground_truth_tensor == 0).sum().item()
    total_positive = true_positive + false_negative
    total_negative = true_negative + false_positive
    TPR = true_positive / total_positive if total_positive > 0 else 0
    FPR = false_positive / total_negative if total_negative > 0 else 0
    return FPR, TPR


class HardPreds:
    """
    For a given attack instance' MIAScore (soft prediction), we use HardPreds to store the hard predictions
    of this score at all FPR values/thresholds. It's used to calculate the AUC and accuracy of the hard predictions' ensemble.
    We also save the ensemble-ed hard predictions at different FPR values.
    """
    def __init__(self, fprs, tprs, hard_preds, gt, name: str):
        """
        Initialize the HardPreds object.

        :param fprs: false positive rates
        :param tprs: true positive rates
        :param name: name of the attack
        :param hard_preds: hard predictions at different FPR values
        """
        self._fprs = fprs
        self._tprs = tprs
        self._hard_preds = hard_preds
        self.gt = gt
        self.name = name
    

    @classmethod
    def from_pred(cls, mia_score: Predictions, name: str = None,
                  fprs_to_align=np.logspace(-6, 0, num=5000)
                  ):
        """
        Initialize the HardPreds object from Predictions object (as mia_score).

        :param mia_score: membership inference attack scores as a Predictions object
        :param name: Optional name for the HardPreds object
        :param fprs_to_align: FPR values to align the hard predictions
        :return: HardPreds object
        """
        if mia_score.is_hard():
            raise ValueError("The predictions should not be hard labels.")

        scores_arr = mia_score.pred_arr
        gt_arr = mia_score.ground_truth_arr
        fprs, tprs, thresholds = roc_curve(gt_arr, scores_arr)

        hard_preds = []
        # align the hard predictions at different FPR values
        for fpr in fprs_to_align:
            idx = np.where(fprs <= fpr)[0][-1]
            threshold = thresholds[idx]
            hard_pred = (scores_arr >= threshold).astype(int)
            hard_preds.append(hard_pred)

        # recalculating fprs and tprs for the hard predictions
        fprs, tprs = [], []
        for p in hard_preds:
            fpr, tpr= get_fpr_tpr_hard_label(p, gt_arr)
            fprs.append(fpr)
            tprs.append(tpr)

        name = mia_score.name if name is None else name
        return cls(fprs, tprs, hard_preds, mia_score.ground_truth_arr, name)
    

    def get_all_preds(self) -> List[Predictions]:
        """
        Get all hard predictions at different FPR values.

        :return: list of hard predictions as Predictions objects
        """
        return [Predictions(hard_pred, self.mia_score.ground_truth_arr, self.name + f"_FPR_{fpr:.6f}")
                for hard_pred, fpr in zip(self._hard_preds, self._fprs)]
    
    def get_all_preds_arr(self) -> List[np.ndarray]:
        """
        Get all hard predictions at different FPR values.

        :return: list of hard predictions as numpy arrays
        """
        return self._hard_preds
    
    
    @classmethod
    def ensemble(cls, hard_preds_list: List['HardPreds'], method: str, name=None) -> 'HardPreds':
        """
        Ensemble the hard predictions from different FPR values.

        :param hard_preds_list: list of HardPreds
        :param method: method for ensemble the predictions: ["union", "intersection", "majority voting"]
        :return: ensemble prediction
        """

        #  ----- check hard_preds_list quality
        if len(hard_preds_list) < 2:
            raise ValueError("At least 2 hard predictions are required for ensemble.")
        
        num_inferred_samples = len(hard_preds_list[0].get_all_preds_arr()[0])
        for hard_preds in hard_preds_list:
            if len(hard_preds.get_all_preds_arr()[0]) != num_inferred_samples:
                raise ValueError("All hard predictions should have the same number of samples.")
        
        # calculate the sets of hard predictions are being ensemble-ed. each set of 
        # hard predictions should have similar fprs.
        set_of_hard_pred_count = len(hard_preds_list[0]._fprs)
        for hard_preds in hard_preds_list:
            if len(hard_preds._fprs) != set_of_hard_pred_count:
                raise ValueError("All hard predictions should have the same number of hard prediction arrays.")
            
        gt = hard_preds_list[0].gt
            
        # ----- ensemble the hard predictions
            
        fprs, tprs, hard_preds, name = [], [], [], name
            
        if method == "union":
            name = hard_preds_list[0].name + "_union" if name is None else name
            for i in range(set_of_hard_pred_count):
                ensemble_pred = np.zeros_like(hard_preds_list[0].get_all_preds_arr()[0])
                for hp in hard_preds_list:
                    ensemble_pred = np.logical_or(ensemble_pred, hp.get_all_preds_arr()[i])
                hard_preds.append(ensemble_pred)

        elif method == "intersection":
            name = hard_preds_list[0].name + "_intersection" if name is None else name
            for i in range(set_of_hard_pred_count):
                ensemble_pred = np.ones_like(hard_preds_list[0].get_all_preds_arr()[0])
                for hp in hard_preds_list:
                    ensemble_pred = np.logical_and(ensemble_pred, hp.get_all_preds_arr()[i])
                hard_preds.append(ensemble_pred)

        elif method == "majority_vote":
            name = hard_preds_list[0].name + "_majority_vote" if name is None else name
            for i in range(set_of_hard_pred_count):
                ensemble_pred = np.zeros_like(hard_preds_list[0].get_all_preds_arr()[0])
                for hard_preds in hard_preds_list:
                    ensemble_pred += hard_preds.get_all_preds_arr()[i]
                ensemble_pred = (ensemble_pred > len(hard_preds_list) / 2).astype(int)
                hard_preds.append(ensemble_pred)

        else:
            raise ValueError("Invalid method for ensemble the hard predictions.")


        # recalculate fprs and tprs for the ensemble-ed hard predictions
        for i in range(set_of_hard_pred_count):
            fpr, tpr = get_fpr_tpr_hard_label(hard_preds[i], gt)
            fprs.append(fpr)
            tprs.append(tpr)
        
        return cls(fprs, tprs, hard_preds, gt, name)
    
    
    def to_pred(self) -> Predictions:
        """
        Get the hard predictions at a specified FPR value.

        :param fpr: FPR value
        :return: hard predictions at the specified FPR value
        """
        return Predictions(self._hard_preds, self.gt, self.name)
    

    def change_name(self, new_name):
        """
        Update the name of the HardPreds object.
        :param new_name: new name of the HardPreds object
        """
        self.name = new_name

    def get_tp(self, count=True) -> np.ndarray:
        """
        Get the indices of the true positive samples.
        """
        tp_lists = []
        for p in self._hard_preds:
            tp_lists.append(np.where((p == 1) & (self.gt == 1))[0])
        if count:
            return [len(tp) for tp in tp_lists]
        
        return tp_lists
    


def plot_roc_hard_preds(hard_preds_list: List[HardPreds], save_dir: str, tp_or_tpr: str = "TP"):
    """
    Plot the ROC curves for the hard predictions from different FPR values.

    Args:
        hard_preds_list (List[HardPreds]): 
        save_dir (str): _description_
    """
    for pred in hard_preds_list:
        if tp_or_tpr == "TP":
            plt.plot(pred._fprs, pred.get_tp(count=True), label=pred.name)
        elif tp_or_tpr == "TPR":
            plt.plot(pred._fprs, pred._tprs, label=pred.name)
    
    if tp_or_tpr == "TPR":
        plt.semilogx()
        plt.semilogy()
        plt.xlim(1e-5, 1)
        plt.ylim(1e-5, 1)
    elif tp_or_tpr == "TP":
        plt.semilogx()
        plt.xlim(1e-5, 1)
        plt.ylim(0, len(pred.get_tp()[0]))
        # set y-axis to log scale
        plt.yscale('log')
    else:
        raise ValueError("Invalid value for tp_or_tpr.")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate") if tp_or_tpr == "TPR" else plt.ylabel("True Positive Count")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    # plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir, format="pdf")

    plt.clf()


    