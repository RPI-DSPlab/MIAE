"""
This script is used to obtain the graphs of MIA experiments on a specific target model, dataset and seeds.
We will have three types of venn diagrams:
1. Venn diagram of the single attack with different seeds
2. Venn diagram of the different attacks with the same seed
3. Venn diagram of the different attacks with the same FPR (false positive rate)

Work flow:
1. Load the predictions of the target model on the dataset for different seeds.
2. Create Predictions Objects.
3. Set the graph parameters: name, save path, etc.
4. Plot and save the graphs.
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


def load_and_create_predictions(attack: List[str], dataset: str, architecture: str, seeds: List[int] = None,
                                data_aug: str = "True") -> Dict[str, List[utils.Predictions]]:
    """
    load the predictions of the attack of all seeds and create the Predictions objects
    :param attack: List[str]: list of attack names
    :param dataset: str: dataset name
    :param architecture: str: target model architecture
    :param seeds: List[int]: list of random seeds
    :param data_aug: str: whether data augmentation is enabled
    :return: Dict[str, List[Predictions]]: dictionary with attack names as keys and corresponding Predictions objects list as values
    """
    if seeds is None:
        seeds = [0]
    if data_aug == "True":
        aug_str = "_aug"
    else:
        aug_str = ""

    # load the target_dataset
    target_dataset_path = f"/data/public/miae_experiment{aug_str}/target/{dataset}/"
    index_to_data, attack_set_membership = utils.load_target_dataset(target_dataset_path)

    pred_dict = {}
    for att in attack:
        pred_list = []
        for s in seeds:
            pred_path = f"/data/public/miae_experiment{aug_str}/preds_sd{s}/{dataset}/{architecture}/{att}/pred_{att}.npy"
            pred_arr = utils.load_predictions(pred_path)
            pred_obj = utils.Predictions(pred_arr, attack_set_membership, att)
            pred_list.append(pred_obj)
        pred_dict[att] = pred_list
    return pred_dict


def plot_venn(pred_dict: Dict[str, List[utils.Predictions]], graph_type: str, graph_title: str, graph_path: str, seed: int = None, fpr: float = None):
    """
    plot the venn diagrams and save them
    :param pred_dict: dictionary with attack names as keys and corresponding Predictions objects list as values
    :param seed: fixed seed used for the experiment
    :param graph_type: type of graph: [single_attack, different_attacks_seed, different_attacks_fpr]
    :param graph_title: title of the graph
    :param graph_path: path to save the graph
    :return: None
    """
    # single_attack: the same attack with different seeds
    if graph_type == "single_attack":
        for att, pred_list in pred_dict.items():
            utils.plot_venn_diagram(pred_list, graph_title, graph_path, graph_type)
    # different_attacks_seed: different attacks with the same seed
    elif graph_type == "different_attacks_seed" and seed is not None:
        matched_pred_list = []
        for att, pred_list in pred_dict.items():
            matched_pred_list.append(pred_list[seed])
        utils.plot_venn_diagram(matched_pred_list, graph_title, graph_path, graph_type)
    # different_attacks_fpr: different attacks with the same FPR
    elif graph_type == "different_attacks_fpr" and fpr is not None and seed is not None:
        matched_pred_list = []
        for att, pred_list in pred_dict.items():
            matched_pred_list.append(pred_list[seed])
        utils.plot_venn_diagram(matched_pred_list, graph_title, graph_path, graph_type)
    else:
        raise ValueError(f"Invalid graph type: {graph_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_graphs')
    # Required arguments
    parser.add_argument("--attacks", type=str, nargs="+", default=None, help='MIA type: [losstraj, yeom, shokri]')
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100, cinic10]')
    parser.add_argument("--architecture", type=str, default="resnet56",
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument("--graph-type", type=str, default="venn", help="Type of graph")
    parser.add_argument("--graph-title", type=str, help="Title of the graph")
    parser.add_argument("--graph-path", type=str, help="Path to save the graph")

    # Optional arguments
    parser.add_argument("--seed", type=int, nargs="+", help="Random seed")
    parser.add_argument("--data_aug", type=str, default="True", help="Whether data augmentation is enabled")

    args = parser.parse_args()

    # load the predictions of the target model on the dataset for different seeds
    pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.seed, args.data_aug)

    # plot and save the graphs
    plot_venn(pred_dict, args.graph_type, args.graph_title, args.graph_path, args.seed[0])




