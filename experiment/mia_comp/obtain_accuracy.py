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
import pandas as pd
import utils
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))
from MIAE.miae.eval_methods.prediction import Predictions, pred_tp_set_op, multi_seed_ensemble

def load_and_create_predictions(attack: List[str], dataset: str, architecture: str, data_path: str, seeds: List[int] = None,
                                ) -> Dict[str, List[Predictions]]:
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
            pred_obj = Predictions(pred_arr, attack_set_membership, attack_name)
            pred_list.append(pred_obj)
        pred_dict[att] = pred_list
    return pred_dict

def data_process(pred_dict: Dict[str, List[Predictions]], process_opts: List[str]):
    """
    Process the predictions based on the process option
    :param pred_dict: Dict[str, List[Predictions]]: dictionary with attack names as keys and corresponding Predictions objects list as values
    :return: Dict[str, List[Predictions]]: dictionary with attack names as keys and corresponding Predictions objects list as values
    """
    pred_dict_processed = {opt: [] for opt in process_opts}

    for attack, pred_list in pred_dict.items():
        pred_union, pred_intersection = pred_tp_set_op(pred_list)
        pred_avg = multi_seed_ensemble(pred_list, "avg")
        pred_dict_processed["union"].append(pred_union)
        pred_dict_processed["intersection"].append(pred_intersection)
        pred_dict_processed["avg"].append(pred_avg)
    return pred_dict_processed

def pearson_correlation_coefficient(pred_list: List[Predictions]):
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

def save_accuracy_results(pred_dict: Dict[str, List[Predictions]], fpr_list: List[float], model: str, dataset: str, file_path: str, process_opt: List[str]):
    """
    Save the accuracy results to a file
    :param pred_dict_processed: Predictions dictionary
    :param fpr_list: List of FPR values
    :param model: Name of the model
    :param dataset: Name of the dataset
    :param file_path: Path to save the results
    :param process_opt: Process option: union or intersection
    """
    pred_dict_processed = data_process(pred_dict, process_opt)
    pred_sd0 = [pred_dict[attack][0] for attack in pred_dict]
    try:
        with open(file_path, "a") as file:
            header = f"Accuracy at seed 0 for each attack under {model} and {dataset}\n"
            line = '-' * (len(header)-1) + '\n'
            file.write(line + header)
            for pred in pred_sd0:
                accuracy_at_sd0 = "{:.2f}".format(pred.accuracy())
                file.write(f"{pred.name.split('_')[0]}: {accuracy_at_sd0}\n")
            file.write(line)

            for opt in process_opt:
                pred_list = pred_dict_processed[opt]
                for pred in pred_list:
                    # header
                    header = f"===== {pred.name} ({opt}) under {model} and {dataset} =====\n"
                    line = '=' * (len(header)-1) + '\n'
                    file.write(f"\n" + line + header + line)

                    # Balanced Accuracy
                    balanced_accuracy = "{:.2f}".format(pred.balanced_attack_accuracy())
                    file.write(f"Balanced Accuracy: {balanced_accuracy}\n")

                    # tpr at fpr
                    for fpr in fpr_list:
                        tpr = "{:.2f}".format(pred.tpr_at_fpr(float(fpr)))
                        file.write(f"TPR at {fpr}% FPR: {tpr}\n")
                    file.write(f"\n")
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
    parser.add_argument("--process_opt", type=str, nargs="+", help="way to process the predictions: [union, intersection]")
    parser.add_argument("--accuracy_path", type=str, help="Path to save the accuracy results")
    parser.add_argument("--data_path", type=str, help="Path to the original predictions and target dataset")
    parser.add_argument("--seeds", type=int, nargs="+", help="Random seed")

    args = parser.parse_args()

    pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path, args.seeds)
    save_accuracy_results(pred_dict, args.fpr_list, args.architecture, args.dataset, args.accuracy_path, args.process_opt)





