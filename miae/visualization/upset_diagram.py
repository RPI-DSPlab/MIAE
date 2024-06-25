import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet
from typing import List, Tuple, Dict, Optional
from miae.eval_methods.prediction import Predictions, union_pred, intersection_pred, find_common_tp_pred, find_common_tn_pred

def create_binary_matrix(attacked_points, labels):
    """
    Create a binary matrix from the attacked points and labels.
    :param attacked_points: dictionary of attacked points for each label
    :param labels: list of labels
    :return: binary matrix
    """
    matrix = np.zeros((len(attacked_points), len(attacked_points)))
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            matrix[i, j] = len(attacked_points[label1].intersection(attacked_points[label2]))
    return pd.DataFrame(matrix, index=labels, columns=labels)

def plot_upset_for_all_attacks(pred_or: List[Predictions], pred_and: List[Predictions], save_path: str):
    """
    Plot Upset diagrams for at most 5 attacks.
    :param pred_or: list of Predictions for the 'pred_or' set
    :param pred_and: list of Predictions for the 'pred_and' set
    :param save_path: path to save the graph
    """
    attacked_points_or = {pred.name: set() for pred in pred_or}
    attacked_points_and = {pred.name: set() for pred in pred_and}

    for pred in pred_or:
        attacked_points_or[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])

    for pred in pred_and:
        attacked_points_and[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])

    df_or = create_binary_matrix(attacked_points_or, [pred.name for pred in pred_or])
    df_and = create_binary_matrix(attacked_points_and, [pred.name for pred in pred_and])

    # Plotting Upset plot for 'or' condition
    plt.figure(figsize=(10, 8))
    upset_or = UpSet(df_or)
    upset_or.plot()
    plt.savefig(f"{save_path}_upset_or.pdf")
    plt.close()

    # Plotting Upset plot for 'and' condition
    plt.figure(figsize=(10, 8))
    upset_and = UpSet(df_and)
    upset_and.plot()
    plt.savefig(f"{save_path}_upset_and.pdf")
    plt.close()