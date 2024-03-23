"""
This script is used to obtain the graphs of MIA experiments on a specific target model, dataset and seeds.
We will have three types of venn diagrams:
1. Venn diagram of the single attack with different seeds  ==> to check how stable the attack is with different seeds
2. Venn diagram of the different attacks with common TP   ==> to compare different attacks with the common TP (true positive)
3. Venn diagram in a pairwise manner                      ==> to compare the attacks in a pairwise manner

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


def plot_venn(pred_list: List[utils.Predictions], graph_goal: str, graph_title: str, graph_path: str):
    """
    plot the venn diagrams and save them
    :param pred_dict: dictionary with attack names as keys and corresponding Predictions objects list as values
    :param graph_goal: goal of the venn diagram: "common_TP" or "single attack"
    :param graph_title: title of the graph
    :param graph_path: path to save the graph
    :return: None
    """
    if graph_goal == "common_tp":
        utils.plot_venn_diagram(pred_list, graph_goal, graph_title, graph_path)
    elif graph_goal == "single_attack":
        utils.plot_venn_diagram(pred_list, graph_goal, graph_title, graph_path)
    elif graph_goal == "pairwise":
        paired_pred_list = utils.find_pairwise_preds(pred_list)
        utils.plot_venn_diagram_pairwise(paired_pred_list, graph_title, graph_path)

def plot_auc(predictions: Dict[str, utils.Predictions], graph_title: str, graph_path: str,
             fprs: List[float] = None, log_scale: bool = True):
    """
    plot the AUC of the different attacks
    :param predictions: List[utils.Predictions]: list of Predictions objects
    :param graph_title: str: title of the graph
    :param graph_path: str: path to save the graph
    :param fprs: List[float]: list of false positive rates to be plotted as vertical lines on auc graph,
    if None, no need to plot any vertical line
    :param log_scale: bool: whether to plot the graph in log scale

    :return: None
    """
    attack_names, prediction_list = [], []
    ground_truth = None
    for attack, pred in predictions.items():
        attack_names.append(attack)
        prediction_list.append(pred.pred_arr)
        ground_truth = pred.ground_truth_arr if ground_truth is None else ground_truth

    utils.plot_auc(prediction_list, attack_names, ground_truth, graph_title, fprs, log_scale, graph_path)


def plot_hardness_distribution(predictions: Dict[str, List[utils.Predictions]] or Dict[str, utils.Predictions],
                               hardness: utils.SampleHardness,
                               graph_title: str, graph_path: str, fpr_list: List[float] = None):
    """
    plot the hardness distribution of the different attacks
    :param predictions: List[utils.Predictions]: list of Predictions objects
    :param graph_title: str: title of the graph
    :param graph_path: str: path to save the graph
    :param hardness: str: type of hardness: [il]
    :return: None
    """
    attack_names, prediction_list = [], []
    for attack, pred in predictions.items():
        attack_names.append(attack)
        prediction_list.append(pred)

        if fpr_list is None:  # prediction is determined by 0.5 threshold
            attack_tp = pred.get_tp()
            hardness.plot_distribution_pred_TP(attack_tp, save_path=graph_path+f"_vs_{attack}_tp.png", title=graph_title+f" {attack} TP")

        else:  # prediction is determined by fpr threshold
            for fpr in fpr_list:
                attack_tp = utils.common_tp(pred, fpr)
                hardness.plot_distribution_pred_TP(attack_tp, save_path=graph_path+f"_vs_{attack}_tp_fpr{fpr}.png", title=graph_title+f" {attack} TP at {fpr} FPR")

    # if fpr_list is None:  # prediction is determined by 0.5 threshold
    #     common_tp = utils.common_tp(prediction_list)
    #     hardness.plot_distribution_pred_TP(common_tp, save_path=graph_path+"_vs_common_tp.png", title=graph_title+" common TP")
    #
    # else:  # prediction is determined by fpr threshold
    #     for fpr in fpr_list:
    #         common_tp = utils.common_tp(prediction_list, fpr)
    #         hardness.plot_distribution_pred_TP(common_tp, save_path=graph_path+f"_vs_common_tp_fpr{fpr}.png", title=graph_title+f" common TP at {fpr} FPR")
    #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_graphs')
    # Required arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100, cinic10]')
    parser.add_argument("--architecture", type=str, default="resnet56",
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument("--attacks", type=str, nargs="+", default=None, help='MIA type: [losstraj, yeom, shokri]')
    parser.add_argument("--graph_type", type=str, default="venn", help="Type of graph")
    parser.add_argument("--graph_title", type=str, help="Title of the graph")
    parser.add_argument("--graph_path", type=str, help="Path to save the graph")
    parser.add_argument("--data_path", type=str, help="Path to the original predictions and target dataset")

    # Optional arguments
    parser.add_argument("--seed", type=int, nargs="+", help="Random seed")

    # graph specific arguments
    # for venn diagram
    parser.add_argument("--graph_goal", type=str, help="Goal of the venn diagram: [common_tp, single_attack]")
    parser.add_argument("--threshold", type=float, help="Threshold for the comparison on venn diagram")
    parser.add_argument("--single_attack_name", type=str, help="Name of the single attack for the venn diagram")

    # for auc graph
    parser.add_argument("--fpr", type=float, nargs="+",
                        help="True positive rate to be plotted as vertical line on auc graph")
    parser.add_argument("--log_scale", type=bool, default="True", help="Whether to plot the graph in log scale")

    # for hardness distribution graph
    parser.add_argument("--hardness", type=str, default="None", help="Type of hardness: [il]")
    parser.add_argument("--hardness_path", type=str, help="Path to the hardness file")

    args = parser.parse_args()

    # load the predictions of the target model on the dataset for different seeds
    pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path, args.seed)

    # plot and save the graphs
    if args.graph_type == "venn":
        if args.graph_goal == "single_attack":
            pred_list = pred_dict[args.single_attack_name][:3]
            plot_venn(pred_list, args.graph_goal, args.graph_title, args.graph_path)
        elif args.graph_goal == "common_tp":
            if args.threshold == 0:
                fpr_list = [float(f) for f in args.fpr]
                for f in fpr_list:
                    pred_list = utils.data_process_for_venn(pred_dict, threshold=0, target_fpr=f)
                    graph_title = args.graph_title+f" FPR = {f}"
                    graph_path = args.graph_path+f"_{f}"
                    plot_venn(pred_list, args.graph_goal, graph_title, graph_path)
            elif args.threshold != 0:
                pred_list = utils.data_process_for_venn(pred_dict, threshold=args.threshold, target_fpr=0)
                graph_title = args.graph_title + f" threshold = {args.threshold}"
                graph_path = args.graph_path + f"_{args.threshold}"
                plot_venn(pred_list, args.graph_goal, graph_title, graph_path)
        elif args.graph_goal == "pairwise":
            if args.threshold == 0:
                fpr_list = [float(f) for f in args.fpr]
                for f in fpr_list:
                    pred_list = utils.data_process_for_venn(pred_dict, threshold=0, target_fpr=f)
                    graph_title = args.graph_title+f" FPR = {f}"
                    graph_path = args.graph_path+f"_{f}"
                    plot_venn(pred_list, args.graph_goal, graph_title, graph_path)
            elif args.threshold != 0:
                pred_list = utils.data_process_for_venn(pred_dict, threshold=args.threshold, target_fpr=0)
                graph_title = args.graph_title + f" threshold = {args.threshold}"
                graph_path = args.graph_path + f"_{args.threshold}"
                plot_venn(pred_list, args.graph_goal, args.graph_title, args.graph_path)
        else:
            raise ValueError(f"Invalid graph goal for Venn Diagram: {args.graph_goal}")
    elif args.graph_type == "auc":
        for i, seed in enumerate(args.seed):
            pred_dict_seed = {k: v[i] for k, v in pred_dict.items()}
            plot_auc(pred_dict_seed, args.graph_title+f" sd{seed}", args.graph_path+f"_sd{seed}.png", args.fpr, args.log_scale)

    elif args.graph_type == "hardness_distribution":
        path_to_load = f"{args.hardness_path}/{args.dataset}/{args.architecture}/{args.hardness}/{args.hardness}_score.pkl"
        hardness_arr = utils.load_example_hardness(path_to_load)
        hardness = utils.SampleHardness(hardness_arr, args.hardness)
        plot_hardness_distribution(pred_dict, hardness, args.graph_title, args.graph_path, args.fpr)

    else:
        raise ValueError(f"Invalid graph type: {args.graph_type}")
