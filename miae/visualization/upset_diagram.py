import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_contents
from typing import List, Dict
from miae.eval_methods.prediction import Predictions


def plot_upset_for_all_attacks(pred_or: List[Predictions], pred_and: List[Predictions], save_path: str):
    """
    Plot Upset diagrams for all attacks.
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

    for attack, points in attacked_points_or.items():
        print(f"Attack: {attack}, Number of attacked points: {len(points)}")

    for attack, points in attacked_points_and.items():
        print(f"Attack: {attack}, Number of attacked points: {len(points)}")

    df_or = from_contents(attacked_points_or)
    df_or = df_or.fillna(False)


    df_and = from_contents(attacked_points_and)
    df_and = df_and.fillna(False)


    # Plotting UpSet plot for 'or' condition
    plt.figure(figsize=(10, 8))
    UpSet(df_or).plot()
    plt.savefig(f"{save_path}_upset_or.pdf")
    plt.close()

    # Plotting UpSet plot for 'and' condition
    plt.figure(figsize=(10, 8))
    UpSet(df_and).plot()
    plt.savefig(f"{save_path}_upset_and.pdf")
    plt.close()
