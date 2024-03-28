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
import numpy as np
import pickle
import utils

import utils
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

def load_and_create_predictions(attack: List[str], dataset: str, architecture: str, data_path: str, seeds: List[int] = None,
                                ) -> Dict[str, List[utils.Predictions]]:
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
            pred_obj = utils.Predictions(pred_arr, attack_set_membership, attack_name)
            pred_list.append(pred_obj)
        pred_dict[att] = pred_list
    return pred_dict

def pearson_correlation_coefficient(pred_list: List[utils.Predictions]):
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

def save_accuracy_results(pred_list: List[utils.Predictions], header: str, file_path: str):
    """
    Save the accuracy results to a file
    :param pred_list: List of Predictions
    :param file_path: Path to save the results
    """
    try:
        with open(file_path, "w") as file:
            file.write(header)
            file.write("\n")

            for pred in pred_list:
                file.write(f"{pred.name}: {pred.accuracy()}")
                file.write(f"{pred.name}: {pred.balanced_attack_accuracy()}")
                file.write("\n")

            file.write(f"=====================\n")
            correlation_dict = pearson_correlation_coefficient(pred_list)
            for key, value in correlation_dict.items():
                file.write(f"The Pearson Correlation Coefficient of {key}: {value}")
    except IOError as e:
        print(f"Error: Unable to write to file '{file_path}': {e}")
    finally:
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_accuracy')
    # Required arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100, cinic10]')
    parser.add_argument("--architecture", type=str, default="resnet56",
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument("--attacks", type=str, nargs="+", default=None, help='MIA type: [losstraj, yeom, shokri]')
    parser.add_argument("--accuracy_path", type=str, help="Path to save the accuracy results")
    parser.add_argument("--data_path", type=str, help="Path to the original predictions and target dataset")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of seeds to consider")

    args = parser.parse_args()

    # load the predictions
    pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path, args.seeds)

    # save the accuracy results
