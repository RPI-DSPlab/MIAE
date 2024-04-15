"""
This file is used to obtain the graphs of MIA experiments on a specific target model, dataset and seeds.
The graphs include: Venn diagrams; AUC graphs; Hardness distribution graphs

Three types of venn diagrams:
1. Venn diagram of the single attack with different seeds  ==> to check how stable the attack is with different seeds
2. Venn diagram of the different attacks with common TP   ==> to compare different attacks with the common TP (true positive)
3. Venn diagram in a pairwise manner                      ==> to compare two attacks
"""
import argparse
import os
import torch
from matplotlib import pyplot as plt
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


def plot_venn(pred_list: List[utils.Predictions], pred_list2: List[utils.Predictions], graph_goal: str, graph_title: str, graph_path: str):
    """
    plot the venn diagrams and save them
    :param pred_dict: dictionary with attack names as keys and corresponding Predictions objects list as values
    :param graph_goal: goal of the venn diagram: "common_TP" or "single attack"
    :param graph_title: title of the graph
    :param graph_path: path to save the graph
    :return: None
    """
    if graph_goal == "common_tp":
        utils.plot_venn_diagram(pred_list, pred_list2, graph_goal, graph_title, graph_path)
    elif graph_goal == "single_attack":
        utils.plot_venn_single(pred_list, graph_title, graph_path)
    elif graph_goal == "pairwise":
        paired_pred_list = utils.find_pairwise_preds(pred_list)
        utils.plot_venn_pairwise(paired_pred_list, graph_title, graph_path)

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


def multi_seed_convergence(predictions: Dict[str, List[utils.Predictions]], graph_title: str, graph_path: str, set_op, attack_fpr=None):
    """
    plot the convergence of the different attacks

    :param predictions: List[utils.Predictions]: list of Predictions objects, each element in a list is a Predictions object for a specific seed
    :param graph_title: str: title of the graph
    :param graph_path: str: path to save the graph
    :param set_op: str: set operation to be used for the convergence: [union, intersection]
    :param attack_fpr: float: false positive rate to be plotted as vertical line on auc graph

    :return: None
    """
    # obtain the number of true positives for each attack at num of seeds
    num_tp_dict = {}
    tpr_dict = {}
    fpr_dict = {}
    precision_dict = {}
    for attack, pred_list in predictions.items():
        num_tp_dict[attack] = []
        tpr_dict[attack] = []
        fpr_dict[attack] = []
        precision_dict[attack] = []
        for i in range(len(pred_list)):
            # agg_tp is the aggregated true positives, agg_pred is the aggregated 1 (member) predictions
            if set_op == "union":
                agg_tp = utils.union_tp(pred_list[:i+1], attack_fpr)
                agg_pred = utils.union_pred(pred_list[:i+1], attack_fpr)
            elif set_op == "intersection":
                agg_tp = utils.intersection_tp(pred_list[:i+1], attack_fpr)
                agg_pred = utils.intersection_pred(pred_list[:i+1], attack_fpr)
            else:
                raise ValueError(f"Invalid set operation: {set_op}")
            num_tp_dict[attack].append(len(agg_tp))

            # -- calculate the true positive rate -- tpr = tp / (tp + fn)
            tp = 0
            gt = pred_list[0].ground_truth_arr
            fn = 0
            for j in range(len(gt)):
                if gt[j] == 1 and j not in agg_pred:
                    fn += 1
                if gt[j] == 1 and j in agg_pred:
                    tp += 1
            tpr = tp / (tp + fn)
            tpr_dict[attack].append(tpr)

            # --- calculate the false positive rate ---  fpr = fp / (fp + tn)
            fp = 0
            tn = 0
            gt = pred_list[0].ground_truth_arr
            for j in range(len(gt)):
                if gt[j] == 0 and j in agg_pred:  # if j is predicted as member and it is not a member from gt
                    fp += 1
                if gt[j] == 0 and j not in agg_pred:  # if j is not predicted as member and it is not a member from gt
                    tn += 1
            fpr = fp / (fp + tn)
            fpr_dict[attack].append(fpr)

            # --- calculate the precision --- precision = tp / (tp + fp)
            precision = tp / (tp + fp) if (tp+fp) != 0 else 0
            precision_dict[attack].append(precision)

    num_plots = 4
    num_seed = 0
    fig, axes = plt.subplots(1, num_plots, figsize=(26, 5))
    fig.subplots_adjust(wspace=0.4)  # Adjust the spacing between subplots
    # Plotting the convergence of number of true positives
    for attack, num_tp in num_tp_dict.items():
        axes[0].plot(num_tp, label=attack)
        num_seed = len(num_tp)
    axes[0].set_xticks(np.arange(num_seed), np.arange(1, num_seed + 1))
    axes[0].set_xlabel("Number of seeds")
    axes[0].set_ylabel("Number of True Positives")
    axes[0].set_title("Number of True Positives Convergence")
    axes[0].legend()

    # Plotting the convergence of true positive rate
    for attack, tpr in tpr_dict.items():
        axes[1].plot(tpr, label=attack)
    axes[1].set_xticks(np.arange(num_seed), np.arange(1, num_seed + 1))
    axes[1].set_xlabel("Number of seeds")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("True Positive Rate Convergence")
    axes[1].legend()

    # Plotting the convergence of false positive rate
    for attack, tpr in fpr_dict.items():
        axes[2].plot(tpr, label=attack)
    axes[2].set_xticks(np.arange(num_seed), np.arange(1, num_seed + 1))
    axes[2].set_xlabel("Number of seeds")
    axes[2].set_ylabel("False Positive Rate")
    axes[2].set_title("False Positive Rate Convergence")
    axes[2].legend()

    # Plotting the convergence of precision
    for attack, precision in precision_dict.items():
        axes[3].plot(precision, label=attack)
    axes[3].set_xticks(np.arange(num_seed), np.arange(1, num_seed + 1))
    axes[3].set_xlabel("Number of seeds")
    axes[3].set_ylabel("Precision")
    axes[3].set_title("Precision Convergence")
    axes[3].legend()

    plt.savefig(graph_path + f"_fpr{attack_fpr}.png", dpi=300)


def single_attack_seed_ensemble(predictions: Dict[str, List[utils.Predictions]], graph_title: str, graph_path: str, num_seeds: int, skip: int=2):
    """
    ensemble attacks from multiple seeds and plot the roc/auc curve for each attack and ensemble method with different number of seeds
    :param predictions: Dict[str, List[utils.Predictions]]: dictionary with attack names as keys and corresponding Predictions objects list as values
    :param graph_title: str: title of the graph
    :param graph_path: str: path to save the graph
    :param num_seeds: int: number of seeds to ensemble
    :param skip: int: number of seeds to skip for each ensemble plotting
    """
    gt_arr = predictions[list(predictions.keys())[0]][0].ground_truth_arr
    for ensemble_method in ["HC", "HP", "avg"]:  # High Coverage and High Precision
        for attack, pred_list in predictions.items():
            ensemble_pred = []
            num_seeds_list = []
            name_list = []
            for i in range(0, num_seeds, skip):
                ensemble_pred.append(utils.multi_seed_ensemble(pred_list[:i+1], ensemble_method, threshold=0.5).pred_arr)
                num_seeds_list.append(i+1)
                name_list.append(attack+f"_{ensemble_method}_numsd{i+1}")

            title = f"{graph_title} {attack} {ensemble_method}"
            print(f"plotting auc for {title}...")
            utils.plot_auc(ensemble_pred, name_list, gt_arr, title, None, True, graph_path+f"_{attack}_{ensemble_method}.png")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_graphs')
    # Required arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100, cinic10]')
    parser.add_argument("--architecture", type=str, default="resnet56",
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument("--attacks", type=str, nargs="+", default=None, help='MIA type: [losstraj, yeom, shokri]')
    parser.add_argument("--graph_type", type=str, default="venn", help="graph_type: [venn, auc, hardness_distribution, multi_seed_convergence, single_attack_seed_ensemble]")
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

    # for single seed ensemble graph
    parser.add_argument("--skip", type=int, default=2, help="Number of seeds to skip for each ensemble plotting")

    args = parser.parse_args()

    # load the predictions of the target model on the dataset for different seeds
    pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path, args.seed)

    # plot and save the graphs
    if args.graph_type == "venn":
        if args.graph_goal == "single_attack":
            pred_list = pred_dict[args.single_attack_name][:3]
            plot_venn(pred_list, [], args.graph_goal, args.graph_title, args.graph_path)
        elif args.graph_goal == "common_tp":
            if args.threshold == 0:
                fpr_list = [float(f) for f in args.fpr]
                for f in fpr_list:
                    pred_or_list, pred_and_list = utils.data_process_for_venn(pred_dict, threshold=0, target_fpr=f)
                    graph_title = args.graph_title+f" FPR = {f}"
                    graph_path = args.graph_path+f"_{f}"
                    plot_venn(pred_or_list, pred_and_list, args.graph_goal, graph_title, graph_path)
            elif args.threshold != 0:
                pred_or_list, pred_and_list = utils.data_process_for_venn(pred_dict, threshold=args.threshold, target_fpr=0)
                graph_title = args.graph_title + f" threshold = {args.threshold}"
                graph_path = args.graph_path + f"_{args.threshold}"
                plot_venn(pred_or_list, pred_and_list, args.graph_goal, graph_title, graph_path)
        elif args.graph_goal == "pairwise":
            if args.threshold == 0:
                fpr_list = [float(f) for f in args.fpr]
                for f in fpr_list:
                    pred_or_list, pred_and_list = utils.data_process_for_venn(pred_dict, threshold=0, target_fpr=f)
                    graph_title = args.graph_title+f" FPR = {f}"
                    graph_path = args.graph_path+f"_{f}"
                    plot_venn(pred_or_list, pred_and_list, args.graph_goal, graph_title, graph_path)
            elif args.threshold != 0:
                pred_or_list, pred_and_list = utils.data_process_for_venn(pred_dict, threshold=args.threshold, target_fpr=0)
                graph_title = args.graph_title + f" threshold = {args.threshold}"
                graph_path = args.graph_path + f"_{args.threshold}"
                plot_venn(pred_or_list, pred_and_list, args.graph_goal, graph_title, graph_path)
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

    elif args.graph_type == "multi_seed_convergence_intersection":
        for fpr in args.fpr:
            graph_title = args.graph_title + f" FPR = {fpr}"
            graph_path = args.graph_path
            multi_seed_convergence(pred_dict, graph_title, graph_path, "intersection", fpr)

    elif args.graph_type == "multi_seed_convergence_union":
        for fpr in args.fpr:
            graph_title = args.graph_title + f" FPR = {fpr}"
            graph_path = args.graph_path
            multi_seed_convergence(pred_dict, graph_title, graph_path, "union", fpr)

    elif args.graph_type == "single_seed_ensemble":
        single_attack_seed_ensemble(pred_dict, args.graph_title, args.graph_path, len(args.seed), skip=args.skip)

    else:
        raise ValueError(f"Invalid graph type: {args.graph_type}")
