"""
This file is used to generate the accuracy of the model

Currently, it supports the following accuracy metrics:
- Balanced Accuracy
- Pearson Correlation Coefficient (When pairwise comparison is used)
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import List, Dict
from tabulate import tabulate
import numpy as np
import pickle

import MIAE.miae.eval_methods.prediction
import utils

import utils
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

def load_and_create_predictions(attack: List[str], dataset: str, architecture: str, data_path: str, seeds: List[int] = None,
                                ) -> Dict[str, List[MIAE.miae.eval_methods.prediction.Predictions]]:
    """
    load the predictions of the attack of all seeds and create the Predictions objects
    :param attack: List[str]: list of attack names
    :param dataset: str: dataset name
    :param architecture: str: target model architecture
    :param seeds: List[int]: list of random seeds
    :return: Dict[str, List[Predictions]]: dictionary with attack names as keys and corresponding Predictions objects list as values
    """

    # load the target_dataset
    target_dataset_path = f"{data_path}/target/{dataset}/"
    index_to_data, attack_set_membership = utils.load_target_dataset(target_dataset_path)

    pred_dict = {}
    for att in attack:
        pred_list = []
        for s in seeds:
            pred_path = f"{data_path}/preds_sd{s}/{dataset}/{architecture}/{att}/pred_{att}.npy"
            pred_arr = utils.load_predictions(pred_path)
            attack_name = f"{att}_sd{s}"
            pred_obj = MIAE.miae.eval_methods.prediction.Predictions(pred_arr, attack_set_membership, attack_name)
            pred_list.append(pred_obj)
        pred_dict[att] = pred_list
    return pred_dict

def data_process(pred_dict: Dict[str, List[MIAE.miae.eval_methods.prediction.Predictions]], process_opt: str):
    """
    Process the predictions based on the process option
    :param pred_dict: Dict[str, List[Predictions]]: dictionary with attack names as keys and corresponding Predictions objects list as values
    :param process_opt: str: process option: union or intersection
    :return: List[Predictions]: List of Predictions
    """
    pred_list = []
    for name, tmp_list in pred_dict.items():
        if process_opt == "union":
            pred_list = MIAE.miae.eval_methods.prediction.union_tp(tmp_list)
        elif process_opt == "intersection":
            pred_list = MIAE.miae.eval_methods.prediction.intersection_tp(tmp_list)
        else:
            raise ValueError(f"Invalid process option: {process_opt}")
    return pred_list

def pearson_correlation_coefficient(pred_list: List[MIAE.miae.eval_methods.prediction.Predictions]):
    """
    Calculate the person correlation coefficient between each pair of predictions
    :param pred_list: List of Predictions
    :return: Person Correlation Coefficient
    """
    correlation_dict = {}
    for i in range(len(pred_list)):
        for j in range(i+1, len(pred_list)):
            correlation = utils.pearson_correlation(pred_list[i].pred_arr, pred_list[j].pred_arr)
            correlation_dict[f"{pred_list[i].name} and {pred_list[j].name}"] = correlation

    return correlation_dict

def save_accuracy_results(pred_list: List[MIAE.miae.eval_methods.prediction.Predictions], fpr_list: List[float], model: str, dataset: str, file_path: str, process_opt: str):
    """
    Save the accuracy results to a file
    :param pred_list: List of Predictions
    :param fpr_list: List of FPR values
    :param dataset: Name of the dataset
    :param file_path: Path to save the results
    :param process_opt: Process option: union or intersection
    """
    try:
        with open(file_path, "w") as file:
            for pred in pred_list:
                file.write(f"The accuracy results for {pred.name} {process_opt} under {model} and {dataset}:\n")
                for fpr in fpr_list:
                    tpr = "{:.2f}".format(pred.tpr_at_fpr(fpr))
                    file.write(f"TPR at {fpr} FPR: {tpr}")
                balanced_accuracy = "{:.2f}".format(pred.balanced_attack_accuracy())
                file.write(f"Balanced Accuracy: {balanced_accuracy}")
            file.write(f"=====================\n")
    except IOError as e:
        print(f"Error: Unable to write to file '{file_path}': {e}")
    finally:
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_accuracy')

    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100]')
    parser.add_argument("--architecture", type=str, default="resnet56",
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument("--attacks", type=str, nargs="+", default=None, help='MIA type: [losstraj, yeom, shokri]')
    parser.add_argument("--fpr_list", type=str, nargs="+", help="fpr values to consider for the accuracy calculation")
    parser.add_argument("--process_opt", type=str, help="way to process the predictions: [union, intersection]")
    parser.add_argument("--accuracy_path", type=str, help="Path to save the accuracy results")
    parser.add_argument("--data_path", type=str, help="Path to the original predictions and target dataset")

    args = parser.parse_args()

    pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path, args.seeds)
    pred_list = data_process(pred_dict, args.process_opt)
    save_accuracy_results(pred_list, args.fpr_list, args.dataset, args.accuracy_path, args.process_opt)



