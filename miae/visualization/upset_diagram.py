import os
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_contents
from typing import List, Dict
from miae.eval_methods.prediction import Predictions
import miae.visualization.venn_diagram as venn_diagram

def data_process_for_upset(pred_or: List[Predictions], pred_and: List[Predictions]):
    """
    Process data for Upset diagrams.
    :param pred_or: list of Predictions for the 'pred_or' set
    :param pred_and: list of Predictions for the 'pred_and' set
    """
    attacked_points_or = {pred.name: set() for pred in pred_or}
    attacked_points_and = {pred.name: set() for pred in pred_and}

    for pred in pred_or:
        attacked_points_or[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])

    for pred in pred_and:
        attacked_points_and[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])

    df_or = from_contents(attacked_points_or)
    df_or = df_or.fillna(False)

    df_and = from_contents(attacked_points_and)
    df_and = df_and.fillna(False)

    return df_or, df_and

def plot_upset(df_or: pd.DataFrame, df_and: pd.DataFrame, save_path: str):
    """
    Plot Upset diagrams for all attacks.
    :param df_or: DataFrame for 'or' condition
    :param df_and: DataFrame for 'and' condition
    :param save_path: path to save the graph
    """
    # Error checking: if all data in the DataFrame is 0, print error message
    if df_or.sum().sum() == 0:
        print("Skip this UpSet Diagram since no point is attacked under union condition")
        return
    if df_and.sum().sum() == 0:
        print("Skip this UpSet Diagram since no point is attacked under union condition")
        return

    # Plotting UpSet plot
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 8))
    UpSet(df_or, sort_by='cardinality', show_counts=True).plot()
    union_save_path = os.path.join(save_path, "upset_union.pdf")
    plt.savefig(union_save_path)
    plt.close()

    # Plotting UpSet plot
    plt.figure(figsize=(10, 8))
    UpSet(df_and, sort_by='cardinality', show_counts=True).plot()
    intersection_save_path = os.path.join(save_path, "upset_intersection.pdf")
    plt.savefig(intersection_save_path)
    plt.close()
