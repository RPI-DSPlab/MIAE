"""
This file is used to obtain the graphs of MIA experiments on a specific target model, dataset and seeds.
The graphs include: Venn diagrams; AUC graphs;

Three types of venn diagrams:
1. Venn diagram of the single attack with different seeds  ==> to check how stable the attack is with different seeds
2. Venn diagram of the different attacks with common TP   ==> to compare different attacks with the common TP (true positive)
3. Venn diagram in a pairwise manner                      ==> to compare two attacks
"""
import argparse
import os
from matplotlib import pyplot as plt
from typing import List, Dict, Optional
import numpy as np

import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

import miae.eval_methods.prediction as prediction
import miae.visualization.venn_diagram as venn_diagram
import miae.eval_methods.prediction
from miae.eval_methods import experiment

# plot settings

COLUMNWIDTH = 241.14749
COLUMNWIDTH_INCH = 0.01384 * COLUMNWIDTH
TEXTWIDTH = 506.295
TEXTWIDTH_INCH = 0.01384 * TEXTWIDTH

mia_name_mapping = {"losstraj": "losstraj", "shokri": "Class-NN", "yeom": "LOSS", "lira": "LiRA", "aug": "aug", "calibration": "loss-cali", "reference": "reference"}
mia_color_mapping = {
    "losstraj": '#1f77b4', "shokri": '#ff7f0e', "yeom": '#2ca02c', "lira": '#d62728', "aug": '#9467bd', "calibration": '#8c564b', "reference": '#e377c2',
    "losstraj": '#1f77b4', "Class-NN": '#ff7f0e', "LOSS": '#2ca02c', "LiRA": '#d62728', "aug": '#9467bd', "loss-cali": '#8c564b', "reference": '#e377c2'
                     }

plt.style.use('seaborn-v0_8-paper')
# set fontsize
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=7)
plt.rc('font', size=7)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)



def load_and_create_predictions(attack: List[str], dataset: str, architecture: str, data_path: str,
                                seeds: List[int] = None,
                                ) -> Dict[str, List[prediction.Predictions]]:
    """
    Load the predictions of the attack of all seeds and create the Predictions objects
    :param attack: List[str]: list of attack names
    :param dataset: str: dataset name
    :param architecture: str: target model architecture
    :param seeds: List[int]: list of random seeds
    :return: Dict[str, List[Predictions]]: dictionary with attack names as keys and corresponding Predictions objects list as values
    """

    # Mapping of old attack names to new attack names
    name_mapping = {
        "losstraj": "losstraj",
        "shokri": "Class-NN",
        "yeom": "LOSS",
        "lira": "LiRA",
        "aug": "aug",
        "calibration": "loss-cali",
        "reference": "reference"
    }

    # Load the target_dataset
    target_dataset_path = f"{data_path}/target/{dataset}/"
    index_to_data, attack_set_membership = experiment.load_target_dataset(target_dataset_path)


    pred_dict = {}
    for att in attack:
        pred_list = []
        for s in seeds:
            att_npy = att

            pred_path = f"{data_path}/preds_sd{s}/{dataset}/{architecture}/{att}/pred_{att_npy}.npy"
            pred_arr = experiment.load_predictions(pred_path)
            new_attack_name = name_mapping.get(att, att)
            if "distribution" in data_path:
                attack_name = f"{new_attack_name}_{dataset}"
            else:
                attack_name = f"{new_attack_name}_{s}"
            pred_obj = prediction.Predictions(pred_arr, attack_set_membership, attack_name)
            pred_list.append(pred_obj)
        if "top" in att:
            pred_dict[attack_name.rsplit('_', 1)[0]] = pred_list
        else:
            pred_dict[attack_name.split('_')[0]] = pred_list

    return pred_dict

def load_diff_distribution(attack_list: List[str], dataset_list: List[str], architecture: str, data_path: str,
                           FPR: float, seeds: List[int] = None, option: str = "TPR") -> Dict[str, List[prediction.Predictions]]:
    """
    Load the predictions of the attack of all seeds in different datasets and create the Predictions objects
    :param attack: List[str]: list of attack names
    :param dataset: List[str]: list of dataset names
    :param architecture: str: target model architecture
    :param data_path: str: path to the original predictions and target dataset
    :param FPR: float: false positive rate
    :param seeds: List[int]: list of random seeds
    :param option: str: option for the comparison on venn diagram, TPR or TNR
    :return: Dict[str, Dict[str, List[Predictions]]]: [attack names, [Predictions objects under different datasets]]
    """

    nested_dict = {}
    for dataset in dataset_list:
        nested_dict[dataset] = load_and_create_predictions(attack_list, dataset, architecture, data_path, seeds)

    # process the nested dictionary to get the final dictionary: [dataset, [attack name, [Predictions objects]]]
    processed_dict = {}
    for dataset, pred_dict in nested_dict.items():
        processed_dict[dataset] = {}
        for attack, pred_list in pred_dict.items():
            if option == "TPR":
                pred_or, pred_and = prediction.find_common_tp_pred(pred_list, FPR)
            elif option == "TNR":
                pred_or, pred_and = prediction.find_common_tn_pred(pred_list, FPR)
            pred_and.update_name(f"{attack}")
            processed_dict[dataset][attack] = pred_and

    # process the processed_dict to get the final dictionary: [attack name, [Predictions objects under different datasets]]
    name_mapping = {
        "losstraj": "losstraj",
        "shokri": "Class-NN",
        "yeom": "LOSS",
        "lira": "LiRA",
        "aug": "aug",
        "calibration": "loss-cali",
        "reference": "reference"
    }
    for att in attack_list:
        new_att = name_mapping.get(att, att)
        attack_list[attack_list.index(att)] = new_att

    final_dict = {}
    for attack in attack_list:
        final_dict[attack] = []
        for dataset in dataset_list:
            key_name = f"{attack}_{dataset}"
            final_dict[attack].append(processed_dict[dataset][key_name])

    return final_dict

def plot_venn(pred_list: List[prediction.Predictions], pred_list2: List[prediction.Predictions],
              graph_goal: str, graph_path: str, top_k: bool):
    """
    plot the venn diagrams and save them
    :param pred_list: list of Predictions objects processed using union
    :param pred_list2: list of Predictions objects processed using intersection
    :param graph_goal: goal of the venn diagram: "common_TP" or "single attack"
    :param graph_path: path to save the graph
    :return: None
    """
    if graph_goal == "common_tp":
        if len(pred_list) == 3:
            venn_diagram.plot_venn_diagram(pred_list, pred_list2, graph_path, signal=False, top_k=top_k)
        else:
            venn_diagram.plot_venn_for_all_attacks(pred_list, pred_list2, graph_path)
    elif graph_goal == "single_attack":
        venn_diagram.plot_venn_single(pred_list, graph_path)
    elif graph_goal == "pairwise":
        paired_pred_list_or = venn_diagram.find_pairwise_preds(pred_list)
        paired_pred_list_and = venn_diagram.find_pairwise_preds(pred_list2)
        venn_diagram.plot_venn_pairwise(paired_pred_list_or, paired_pred_list_and, graph_path, top_k=top_k)
    elif graph_goal == "dif_distribution":
        venn_diagram.plot_venn_single(pred_list, graph_path)


def eval_metrics(pred_list: List[prediction.Predictions], save_path: str, title: str, process: Optional[str]):
    """
    Calculate the evaluation metrics for the given list of Predictions.
    :param pred_list: list of Predictions of all attacks
    :param save_path: path to save the metrics
    :return: dictionary of evaluation metrics
    """
    pairwise_jaccard = venn_diagram.pairwise_jaccard_similarity(pred_list)
    overall_jaccard = venn_diagram.overall_jaccard_similarity(pred_list)
    pairwise_overlap_coeff = venn_diagram.pairwise_overlap_coefficient(pred_list)
    overlap_coeff = venn_diagram.overall_overlap_coefficient(pred_list)
    set_size_var = venn_diagram.set_size_variance(pred_list)
    ent = venn_diagram.entropy(pred_list)

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Construct the filename
    filename = os.path.join(save_path, os.path.basename(save_path) + ".txt")

    # Open the file for appending in the directory
    with open(filename, "a") as f:
        f.write("\n")
        if process != "None":
            f.write(f"{title} [processed using {process}]\n")
        else:
            f.write(f"{title}\n")

        f.write("(1) Pairwise Jaccard Similarity\n")
        for result in pairwise_jaccard:
            pair = result[0]
            f.write(f"    {pair[0].name} vs {pair[1].name}: {result[1]:.4f}\n")
        f.write(f"(2) Average Jaccard Similarity: {overall_jaccard:.4f}\n")
        f.write("(3) Pairwise Overlap Coefficient\n")
        for result in pairwise_overlap_coeff:
            pair = result[0]
            f.write(f"    {pair[0].name} vs {pair[1].name}: {result[1]:.4f}\n")
        f.write(f"(4) Average Overlap Coefficient: {overlap_coeff:.4f}\n")
        f.write(f"(5) Set Size Variance: {set_size_var:.4f}\n")
        f.write(f"(6) Entropy: {ent:.4f}\n")
        f.write("\n")


def plot_auc(predictions: Dict[str, prediction.Predictions], graph_title: str, graph_path: str,
             fprs: List[float] = None):
    """
    plot the AUC of the different attacks
    :param predictions: List[Predictions]: list of Predictions objects
    :param graph_title: str: title of the graph
    :param graph_path: str: path to save the graph
    :param fprs: List[float]: list of false positive rates to be plotted as vertical lines on auc graph,
    if None, no need to plot any vertical line

    :return: None
    """
    attack_names, prediction_list = [], []
    ground_truth = None
    for attack, pred in predictions.items():
        attack_names.append(attack)
        prediction_list.append(pred)
        ground_truth = pred.ground_truth_arr if ground_truth is None else ground_truth

    prediction.plot_auc(prediction_list, attack_names, graph_title, fprs, graph_path)


def multi_seed_convergence(predictions: Dict[str, List[prediction.Predictions]], graph_title: str, graph_path: str,
                           set_op, attack_fpr=None):
    """
    plot the convergence of the different attacks

    :param predictions: List[Predictions]: list of Predictions objects, each element in a list is a Predictions object for a specific seed
    :param graph_title: str: title of the graph
    :param graph_path: str: path to save the graph
    :param set_op: str: set operation to be used for the convergence: [union, intersection]
    :param attack_fpr: float: fpr of the each attack to be aggregated from

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
                agg_tp = prediction.union_tp(pred_list[:i + 1], attack_fpr)
                agg_pred = prediction.union_pred(pred_list[:i + 1], attack_fpr)
            elif set_op == "intersection":
                agg_tp = prediction.intersection_tp(pred_list[:i + 1], attack_fpr)
                agg_pred = prediction.intersection_pred(pred_list[:i + 1], attack_fpr)
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
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            precision_dict[attack].append(precision)

    def plot_convergence(y_dict, num_seed, y_axis_option: str, save_dir):
        """
        plot the convergence of the different attacks
        """

        plt.figure(figsize=(COLUMNWIDTH_INCH, 3))
        # setting font size
        plt.rcParams.update({'font.size': 7})
        plt.rcParams["font.weight"] = "bold"

        for idx, (attack, y) in enumerate(y_dict.items()):  # y here is the data to plot on the y-axis
            color = mia_color_mapping[attack]
            plt.plot(y, label=attack, color=color, markerfacecolor='none')
            # annotate the last point
            if y_axis_option == "Number of True Positives":  # integer values
                plt.annotate(f"{int(y[-1])}", (num_seed - 1, y[-1]), textcoords="offset points", xytext=(0, 5),
                             ha='center', fontsize=7, color=color)
            else:
                plt.annotate(f"{y[-1]:.2f}", (num_seed - 1, y[-1]), textcoords="offset points", xytext=(0, 5),
                             ha='center', fontsize=7, color=color)

        plt.xticks(np.arange(num_seed), np.arange(1, num_seed + 1))
        plt.xlabel("Number of instances", fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir, format='pdf')
        plt.close()

    num_seed = len(predictions[list(predictions.keys())[0]])

    for y_dict, y_axis_option in [(num_tp_dict, "Number of True Positives"), (tpr_dict, "True Positive Rate"),
                                  (fpr_dict, "False Positive Rate"), (precision_dict, "Precision")]:
        if os.path.exists(f"{graph_path}/{attack_fpr}") is False:
            os.makedirs(f"{graph_path}/{attack_fpr}")
        save_dir = f"{graph_path}/{attack_fpr}/{y_axis_option.replace(' ', '_')}.pdf"
        plot_convergence(y_dict, num_seed, y_axis_option, save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_graphs')
    # Required arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100, cinic10]')
    parser.add_argument("--architecture", type=str, default="resnet56",
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument("--attacks", type=str, nargs="+", default=None, help='MIA type: [losstraj, yeom, shokri]')
    parser.add_argument("--graph_type", type=str, default="venn",
                        help="graph_type: [venn, auc, multi_seed_convergence]")
    parser.add_argument("--graph_title", type=str, help="Title of the graph")
    parser.add_argument("--graph_path", type=str, help="Path to save the graph")
    parser.add_argument("--data_path", type=str, help="Path to the original predictions and target dataset")

    # Optional arguments
    parser.add_argument("--seed", type=int, nargs="+", help="Random seed")

    # graph specific arguments
    # for venn diagram
    parser.add_argument("--graph_goal", type=str, help="Goal of the venn diagram: [common_tp, single_attack]")
    parser.add_argument("--threshold", type=float, help="Threshold for the comparison on venn diagram")
    parser.add_argument("--FPR", type=float, help="FPR for the comparison on venn diagram")
    parser.add_argument("--single_attack_name", type=str, help="Name of the single attack for the venn diagram")
    parser.add_argument("--dataset_list", type=str, nargs="+", help="all datasets to be compared")
    parser.add_argument("--option", type=str, help="Option for the comparison on venn diagram, TPR or TNR")
    parser.add_argument("--top_k", type=int, help="Top k to be considered for the comparison on venn diagram")

    # for auc graph
    parser.add_argument("--fpr_for_auc", action='store_true',
                        help="True positive rate to be plotted as vertical line on auc graph")
    parser.add_argument("--log_scale", type=bool, default="True", help="Whether to plot the graph in log scale")

    # for convergence graph
    parser.add_argument("--fpr", type=float, nargs="+",
                        help="fprs of instances for convergence graph (coverage/stability) to be aggregating from")

    # for single seed ensemble graph
    parser.add_argument("--skip", type=int, default=0, help="Number of seeds to skip for each ensemble plotting")

    args = parser.parse_args()

    # load the predictions of the target model on the dataset for different seeds
    name_mapping = {
        "losstraj": "losstraj",
        "shokri": "Class-NN",
        "yeom": "LOSS",
        "lira": "LiRA",
        "aug": "aug",
        "calibration": "loss-cali",
        "reference": "reference"
    }
    args.single_attack_name = name_mapping.get(args.single_attack_name, args.single_attack_name)

    # Deal with Top_k
    topK = None
    if args.top_k == 0:
        topK = False
    elif args.top_k == 1:
        topK = True
        print(f"top_k is set to True")

    if args.graph_type == "venn" and len(args.seed) > 1:
        if args.graph_goal == "single_attack":
            pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                    args.seed)
            target_pred_list = pred_dict[args.single_attack_name]
            new_attack_name = name_mapping.get(args.single_attack_name, args.single_attack_name)
            adjusted_pred_list = venn_diagram.single_attack_process_for_venn(target_pred_list, args.FPR)

            graph_title = args.graph_title.replace(args.single_attack_name, new_attack_name)
            graph_path = args.graph_path
            plot_venn(adjusted_pred_list, [], args.graph_goal, graph_path, top_k=topK)
            eval_metrics(adjusted_pred_list, graph_path, graph_title, "None")
        elif args.graph_goal == "common_tp":
            pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                    args.seed)
            if args.threshold == 0: # FPR
                pred_or_list, pred_and_list = venn_diagram.data_process_for_venn(pred_dict, threshold=0,
                                                                                 target_fpr=args.FPR, option=args.option)
                plot_venn(pred_or_list, pred_and_list, args.graph_goal, args.graph_path, top_k=topK)
                pairwise_jaccard_or = eval_metrics(pred_or_list, args.graph_path, args.graph_title, "union")
                pairwise_jaccard_and = eval_metrics(pred_and_list, args.graph_path, args.graph_title, "intersection")
            elif args.threshold != 0: # Threshold
                pred_or_list, pred_and_list = venn_diagram.data_process_for_venn(pred_dict, threshold=args.threshold,
                                                                                 target_fpr=None, option=args.option)
                plot_venn(pred_or_list, pred_and_list, args.graph_goal, args.graph_path)
                eval_metrics(pred_or_list, args.graph_path, args.graph_title, "union")
                eval_metrics(pred_and_list, args.graph_path, args.graph_title, "intersection")
        elif args.graph_goal == "pairwise":
            pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                    args.seed)
            if args.threshold == 0:
                pred_or_list, pred_and_list = venn_diagram.data_process_for_venn(pred_dict, threshold=0,
                                                                                 target_fpr=args.FPR, option=args.option)
                plot_venn(pred_or_list, pred_and_list, args.graph_goal, args.graph_path, top_k=topK)
                eval_metrics(pred_or_list, args.graph_path, args.graph_title, "union")
                eval_metrics(pred_and_list, args.graph_path, args.graph_title, "intersection")
            elif args.threshold != 0:
                pred_or_list, pred_and_list = venn_diagram.data_process_for_venn(pred_dict, threshold=args.threshold,
                                                                                 target_fpr=None, option=args.option)
                plot_venn(pred_or_list, pred_and_list, args.graph_goal, args.graph_path, top_k=topK)
                eval_metrics(pred_or_list, args.graph_path, args.graph_title, "union")
                eval_metrics(pred_and_list, args.graph_path, args.graph_title, "intersection")
        elif args.graph_goal == "dif_distribution":
            attack_list = [args.single_attack_name]
            pred_dict = load_diff_distribution(attack_list, args.dataset_list, args.architecture, args.data_path, args.FPR, args.seed, args.option)
            for attack, pred_list in pred_dict.items():
                plot_venn(pred_list, [], args.graph_goal, args.graph_path, top_k=topK)
        elif args.graph_goal == "model_compare":
            pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                    args.seed)
            venn_diagram.compare_models(pred_dict, args.FPR, args.architecture, args.graph_path)
        else:
            raise ValueError(f"Invalid graph goal for Venn Diagram: {args.graph_goal}")

    elif args.graph_type == "venn" and len(args.seed) == 1:
       if args.graph_goal == "common_tp":
            pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                    args.seed)
            pred_list = venn_diagram.single_seed_process_for_venn(pred_dict, threshold=args.threshold,
                                                                  target_fpr=args.FPR)
            plot_venn(pred_list, pred_list, args.graph_goal, args.graph_path, top_k=topK)
            eval_metrics(pred_list, args.graph_path, args.graph_title, "union")
            eval_metrics(pred_list, args.graph_path, args.graph_title, "intersection")
       elif args.graph_goal == "pairwise":
            pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                    args.seed)
            pred_list = venn_diagram.single_seed_process_for_venn(pred_dict, threshold=args.threshold,
                                                                  target_fpr=args.FPR)
            plot_venn(pred_list, pred_list, args.graph_goal, args.graph_path, top_k=topK)
            eval_metrics(pred_list, args.graph_path, args.graph_title, "union")
            eval_metrics(pred_list, args.graph_path, args.graph_title, "intersection")
       elif args.graph_goal == "dif_distribution":
            attack_list = [args.single_attack_name]
            pred_dict = load_diff_distribution(attack_list, args.dataset_list, args.architecture, args.data_path,
                                               args.FPR, args.seed, args.option)
            for attack, pred_list in pred_dict.items():
                plot_venn(pred_list, [], args.graph_goal, args.graph_path, top_k=topK)
       else:
            raise ValueError(f"Invalid graph goal for Venn Diagram: {args.graph_goal}")

    elif args.graph_type == "auc":
        pred_dict = load_diff_distribution(args.attacks, args.dataset_list, args.architecture, args.data_path, args.FPR,
                                           args.seed)
        for i, seed in enumerate(args.seed):
            pred_dict_seed = {k: v[i] for k, v in pred_dict.items()}
            if args.fpr_for_auc:
                plot_auc(pred_dict_seed, args.graph_title + f" sd{seed}", args.graph_path + f"_sd{seed}.pdf", args.fpr)
            else:
                plot_auc(pred_dict_seed, args.graph_title + f" sd{seed}", args.graph_path + f"_sd{seed}.pdf", None)

    elif args.graph_type == "multi_seed_convergence_intersection":
        pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                args.seed)
        for fpr in args.fpr:
            graph_title = args.graph_title + f" FPR = {fpr}"
            graph_path = args.graph_path
            multi_seed_convergence(pred_dict, graph_title, graph_path, "intersection", fpr)

    elif args.graph_type == "multi_seed_convergence_union":
        pred_dict = load_and_create_predictions(args.attacks, args.dataset, args.architecture, args.data_path,
                                                args.seed)
        for fpr in args.fpr:
            graph_title = args.graph_title + f" FPR = {fpr}"
            graph_path = args.graph_path
            multi_seed_convergence(pred_dict, graph_title, graph_path, "union", fpr)

    else:
        raise ValueError(f"Invalid graph type: {args.graph_type}")
