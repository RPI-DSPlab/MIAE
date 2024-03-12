import os
import pickle
from typing import List, Optional, Tuple, Callable, Union

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


def sweep(score: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute a Receiver Operating Characteristic (ROC) curve.

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


def do_plot(prediction: np.ndarray,
            answers: np.ndarray,
            legend: str = '',
            sweep_fn: Callable = sweep,
            **plot_kwargs: Union[int, str, float]) -> Tuple[float, float]:
    """
    Generate the ROC curves.

    Args:
        prediction (np.ndarray): The predicted scores.
        answers (np.ndarray): The ground truth labels.
        legend (str, optional): Legend for the plot. Defaults to ''.
        sweep_fn (Callable, optional): Function used to compute the ROC curve. Defaults to sweep.

    Returns:
        Tuple[float, float]: Accuracy and Area Under the Curve (AUC).
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < .001)[0][-1]] if np.any(fpr < .001) else 0

    print(f'Attack: {legend.strip():<20} AUC: {auc:<8.4f} max Accuracy: {acc:<8.4f} TPR@0.1%FPR: {low:<8.4f}')

    metric_text = f'auc={auc:.3f}'

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)

    return acc, auc


def fig_fpr_tpr(predictions: List[np.ndarray],
                answers: List[np.ndarray],
                legends: List[str],
                save_dir: Optional[str] = None,
                title: str = "ROC AUC Plot") -> None:
    """
    Function to plot the ROC curve for the given predictions and answers.
    It can handle multiple predictions and answers and assign them different
    legends.

    Args:
        predictions (List[np.ndarray]): List of prediction arrays.
        answers (List[np.ndarray]): Corresponding list of true answers arrays.
        legends (List[str]): List of legends for each prediction-answer pair.
        save_dir (str, optional): Directory where to save the plot. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "ROC AUC Plot".

    Returns:
        None
    """
    plt.figure(figsize=(6, 5))
    plt.title(title)

    for prediction, answer, legend in zip(predictions, answers, legends):
        do_plot(prediction, answer,
                f"{legend}\n")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    # plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir)

    plt.show()

