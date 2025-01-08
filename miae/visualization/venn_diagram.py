
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from venn import venn
import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted, venn3, venn2_unweighted, venn2
from typing import List, Tuple, Dict, Optional
from miae.eval_methods.prediction import Predictions, find_common_tp_pred, find_common_tn_pred


def find_pairwise_preds(pred_list: List[Predictions]) -> List[Tuple[Predictions, Predictions]]:
    """
    Find all possible pairs of predictions in the given list.
    :param pred_list: list of Predictions
    :return: list of tuples, each containing a pair of Predictions
    """
    pairs = []
    n = len(pred_list)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((pred_list[i], pred_list[j]))
    return pairs

def jaccard_similarity(pred_1: Predictions, pred_2: Predictions) -> float:
    """
    Calculate the Jaccard similarity between two predictions.
    :param pred_1: Predictions object
    :param pred_2: Predictions object
    :return: Jaccard similarity
    """
    attacked_points_1 = set(np.where((pred_1.pred_arr == 1) & (pred_1.ground_truth_arr == 1))[0])
    attacked_points_2 = set(np.where((pred_2.pred_arr == 1) & (pred_2.ground_truth_arr == 1))[0])
    intersection = len(attacked_points_1.intersection(attacked_points_2))
    union = len(attacked_points_1.union(attacked_points_2))
    return intersection / union if union != 0 else 0

def pairwise_jaccard_similarity(pred_list: List[Predictions]):
    """
    Calculate the pairwise Jaccard similarity between all pairs of predictions in the given list.
    :param pred_list: list of Predictions
    :return: list of pairwise Jaccard similarities
    """
    pairs = find_pairwise_preds(pred_list)
    pairwise_jaccard = []
    for pair in pairs:
        sim = jaccard_similarity(pair[0], pair[1])
        pairwise_jaccard.append((pair, sim))
    return pairwise_jaccard

def overall_jaccard_similarity(pred_list: List[Predictions]) -> float:
    """
    Calculate the overall Jaccard similarity between all pairs of predictions in the given list.
    :param pred_list: list of Predictions
    :return: overall Jaccard similarity
    """
    pairwise_jaccard = pairwise_jaccard_similarity(pred_list)
    all_sim = [sim for _, sim in pairwise_jaccard]
    return np.mean(all_sim) if all_sim else 0

def overlap_coefficient(pred_1: Predictions, pred_2: Predictions) -> float:
    """
    Calculate the overlap coefficient between two predictions.
    :param pred_1: Predictions object
    :param pred_2: Predictions object
    :return: overlap coefficient
    """
    attacked_points_1 = set(np.where((pred_1.pred_arr == 1) & (pred_1.ground_truth_arr == 1))[0])
    attacked_points_2 = set(np.where((pred_2.pred_arr == 1) & (pred_2.ground_truth_arr == 1))[0])
    intersection = len(attacked_points_1.intersection(attacked_points_2))
    min_size = min(len(attacked_points_1), len(attacked_points_2))
    return intersection / min_size if min_size != 0 else 0

def pairwise_overlap_coefficient(pred_list: List[Predictions]):
    """
    Calculate the pairwise overlap coefficient between all pairs of predictions in the given list.
    :param pred_list: list of Predictions
    :return: list of pairwise overlap coefficients
    """
    pairs = find_pairwise_preds(pred_list)
    pairwise_overlap = []
    for pair in pairs:
        sim = overlap_coefficient(pair[0], pair[1])
        pairwise_overlap.append((pair, sim))
    return pairwise_overlap

def overall_overlap_coefficient(pred_list: List[Predictions]) -> float:
    """
    Calculate the overall overlap coefficient between all pairs of predictions in the given list.
    :param pred_list: list of Predictions
    :return: overall overlap coefficient
    """
    pairwise_overlap = pairwise_overlap_coefficient(pred_list)
    all_sim = [sim for _, sim in pairwise_overlap]
    return np.mean(all_sim) if all_sim else 0

def set_size_variance(pred_list: List[Predictions]) -> float:
    """
    Calculate the variance of the size of the attacked points set.
    :param pred_list: list of Predictions
    :return: variance of the size of the attacked points set
    """
    attacked_points = [set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0]) for pred in pred_list]
    attacked_points_size = [len(points) for points in attacked_points]
    return np.var(attacked_points_size)

def entropy(pred_list: List[Predictions]) -> float:
    """
    Calculate the entropy of the attacked points set.
    :param pred_list: list of Predictions
    :return: entropy of the attacked points set
    """
    attacked_points = [set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0]) for pred in pred_list]
    flattened_attacked_points = [point for points in attacked_points for point in points]
    unique_points, counts = np.unique(flattened_attacked_points, return_counts=True)
    probs = counts / len(flattened_attacked_points)
    return -np.sum(probs * np.log2(probs))

def single_seed_process_for_venn(pred_dict: Dict[str, List[Predictions]], threshold: Optional[float] = 0,
                          target_fpr: Optional[float] = 0) -> List[Predictions]:
    """
    Process the data for the Venn diagram with only one seed: get the pred_list
    :param pred_dict: dictionary of Predictions from different attacks, key: attack name, value: list of Predictions with a single seed
    :param threshold: threshold for the comparison (only used when the graph is generated by threshold otherwise None)
    :param target_fpr: target FPR for the comparison (only used when the graph is generated by FPR otherwise None)
    :return: list of Predictions objects
    """
    if len(pred_dict) < 2:
        raise ValueError("There is not enough data for comparison.")

    if threshold != 0:
        result = []
        for attack, pred_obj_list in pred_dict.items():
            pred = Predictions(pred_obj_list[0].pred_arr, pred_obj_list[0].ground_truth_arr, attack)
            result.append(pred)
    else:
        result = []
        for attack, pred_obj_list in pred_dict.items():
            adjusted_pred_arr = pred_obj_list[0].adjust_fpr(target_fpr)
            name = pred_obj_list[0].name.rsplit('_', 1)[0]
            adjusted_pred_obj = Predictions(adjusted_pred_arr, pred_obj_list[0].ground_truth_arr, name)
            result.append(adjusted_pred_obj)

    return result

def single_attack_process_for_venn(pred_list: List[Predictions], target_fpr: float = 0) -> List[Predictions]:
    """
    Process the data for the Venn diagram with only one attack: get the pred_list
    :param pred_list: list of Predictions from different seeds
    :param target_fpr: target FPR for the comparison (only used when the graph is generated by FPR otherwise None)
    :return: list of Predictions objects
    """
    if target_fpr != None:
        result = []
        for pred in pred_list:
            adjusted_pred_arr = pred.adjust_fpr(target_fpr)
            adjusted_pred_obj = Predictions(adjusted_pred_arr, pred.ground_truth_arr, pred.name)
            result.append(adjusted_pred_obj)
    else:
        raise ValueError("Target_fpr should be provided.")
    return result


def plot_venn_single(pred_list: List[Predictions], save_path: str):
    """
    Plot Venn diagrams for a single attack with 3 different seeds including both unweighted and weighted Venn diagrams.
    :param pred_list: list of Predictions objects
    :param save_path: path to save the graphs
    """
    attacked_points = {pred.name: set() for pred in pred_list}
    venn_sets = []
    for pred in pred_list:
        attacked_points[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0].tolist())
        venn_sets.append(attacked_points[pred.name])


    # venn_sets = tuple(attacked_points[pred.name] for pred in pred_list)
    venn_labels = [pred.name for pred in pred_list]
    venn_unweighted = venn3_unweighted if len(pred_list) == 3 else venn2_unweighted
    venn_weighted = venn3 if len(pred_list) == 3 else venn2

    # Color mapping
    plt.style.use('seaborn-v0_8-paper')
    mia_color_mapping = {
        "losstraj": '#1f77b4', "Class-NN": '#ff7f0e', "LOSS": '#2ca02c',
        "LiRA": '#d62728', "aug": '#9467bd', "loss-cali": '#8c564b',
        "reference": '#e377c2'
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    circle_colors = [mia_color_mapping.get(pred.name, default_colors[i % len(default_colors)]) for i, pred in
                     enumerate(pred_list)]

    # subset_sizes = [len(s) for s in venn_sets]
    # total = sum(subset_sizes)
    total = len(set.union(*venn_sets))
    if total == 0:
        print("Skip plotting because all sets are empty.")
    else:
        # Plotting unweighted Venn diagram
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.axis('off')

        venn = venn_unweighted(subsets=venn_sets, set_labels=venn_labels, 
                                set_colors=circle_colors[:len(pred_list)],
                                subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x / total):.1%}" + ")")

        # Adjust text positions and sizes
        for text in venn.set_labels:
            text.set_fontsize(16)
            text.set_fontweight('bold')

        for text in venn.subset_labels:
            if text is not None:
                text.set_fontsize(16)
                text.set_fontweight('bold')

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        fpr = save_path.split('/')[-1].split('_')[-1]
        att = save_path.split('/')[-2]
        file_name = f"venn_{att}_{fpr}_unweighted.pdf"
        full_save_path = os.path.join(save_path, file_name)
        plt.savefig(full_save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Plotting weighted Venn diagram
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.axis('off')

        venn = venn_weighted(subsets=venn_sets, set_labels=venn_labels, 
                            set_colors=circle_colors[:len(pred_list)],
                            subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x / total):.1%}" + ")")

        # Adjust text positions and sizes
        for text in venn.set_labels:
            text.set_fontsize(16)
            text.set_fontweight('bold')

        for text in venn.subset_labels:
            if text is not None:
                text.set_fontsize(16)
                text.set_fontweight('bold')

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        fpr = save_path.split('/')[-1].split('_')[-1]
        att = save_path.split('/')[-2]
        file_name = f"venn_{att}_{fpr}_weighted.pdf"
        full_save_path = os.path.join(save_path, file_name)
        plt.savefig(full_save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

def plot_venn_single_for_all_seeds(pred_list: List[Predictions], graph_title: str, save_path: str):
    """
    Plot Venn diagrams for a single attack with up to 6 different seeds including both unweighted and weighted Venn diagrams.
    :param pred_list: list of Predictions objects
    :param graph_title: title of the graph
    :param save_path: path to save the graphs
    """
    plt.figure(figsize=(20, 10))
    attacked_points = {pred.name: set() for pred in pred_list}
    for pred in pred_list:
        attacked_points[pred.name] = set(np.where((pred.predictions_to_labels() == pred.ground_truth_arr))[0].tolist())

    venn_sets = [attacked_points[pred.name] for pred in pred_list]
    venn_labels = [pred.name for pred in pred_list]
    circle_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
    # Plotting unweighted Venn diagram
    ax1 = plt.subplot(gs[0, 0], aspect='equal')
    venn({label: set_ for label, set_ in zip(venn_labels, venn_sets)}, cmap="cool", fontsize=10, legend_loc="upper left", ax=ax1)
    plt.title("Unweighted", fontsize=15)

    # Plotting weighted Venn diagram
    ax2 = plt.subplot(gs[0, 1], aspect='equal')
    venn({label: set_ for label, set_ in zip(venn_labels, venn_sets)}, cmap="viridis", fontsize=10, legend_loc="upper left", ax=ax2)
    plt.title("Weighted", fontsize=15)

    plt.suptitle(graph_title, fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_path}.png")


def plot_venn_pairwise(pred_pair_list_or: List[Tuple[Predictions, Predictions]],
                       pred_pair_list_and: List[Tuple[Predictions, Predictions]], save_path: str, top_k: bool):
    """
    Plot Venn diagrams for each pair of predictions in the given lists including both unweighted and weighted Venn diagrams.
    :param pred_pair_list_or: list of tuples, each containing a pair of Predictions objects for union of seeds
    :param pred_pair_list_and: list of tuples, each containing a pair of Predictions objects for intersection of seeds
    :param save_path: path to save the graphs
    :param top_k: whether the top_k is included in the attack
    """
    plt.style.use('seaborn-v0_8-paper')
    mia_color_mapping = {
        "losstraj": '#1f77b4', "Class-NN": '#ff7f0e', "LOSS": '#2ca02c',
        "LiRA": '#d62728', "aug": '#9467bd', "loss-cali": '#8c564b',
        "reference": '#e377c2'
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    def pairwise(pred_1, pred_2, weighted, suffix, folder_path, top_k):
        attacked_points_1 = set(np.where((pred_1.pred_arr == 1) & (pred_1.ground_truth_arr == 1))[0])
        attacked_points_2 = set(np.where((pred_2.pred_arr == 1) & (pred_2.ground_truth_arr == 1))[0])

        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.axis('off')

        if top_k:
            # name1 = pred_1.name.rsplit('_', 1)[0]
            # name2 = pred_2.name.rsplit('_', 1)[0]
            name1 = pred_1.name.split("_")[1] + "_" + pred_1.name.split("_")[2]
            name2 = pred_2.name.split("_")[1] + "_" + pred_2.name.split("_")[2]
        else:
            name1 = pred_1.name.split('_')[0]
            name2 = pred_2.name.split('_')[0]
        circle_colors = [mia_color_mapping.get(name1, default_colors[0]),
                         mia_color_mapping.get(name2, default_colors[1])]

        venn_func = venn2 if weighted else venn2_unweighted

        subset_sizes = [len(attacked_points_1 - attacked_points_2),
                        len(attacked_points_2 - attacked_points_1),
                        len(attacked_points_1 & attacked_points_2)]
        total = sum(subset_sizes)

        venn = venn_func(subsets=(attacked_points_1, attacked_points_2),
                         set_labels=(name1, name2),
                         set_colors=circle_colors,
                         subset_label_formatter=lambda x: f"{x}\n({(x / total):.1%})" if total > 0 else "0\n(0.0%)")

        # Adjust text positions and sizes
        for text in venn.set_labels:
            text.set_fontsize(16)
            text.set_fontweight('bold')

        for text in venn.subset_labels:
            if text is not None:
                text.set_fontsize(16)
                text.set_fontweight('bold')

        plt.tight_layout()
        attacks = folder_path.split('/')[-1]
        fpr = folder_path.split('/')[-2].replace('_', '')
        full_name = f"{attacks}_{fpr}_venn_{suffix}_{'weighted' if weighted else 'unweighted'}.pdf"
        full_save_path = os.path.join(folder_path, full_name)
        plt.savefig(full_save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    for idx, (pair_or, pair_and) in enumerate(zip(pred_pair_list_or, pred_pair_list_and)):
        pred_1_or, pred_2_or = pair_or
        pred_1_and, pred_2_and = pair_and
        if top_k:
            # name1 = pred_1_or.name.rsplit('_', 1)[0]
            # name2 = pred_2_or.name.rsplit('_', 1)[0]
            name1 = pred_1_or.name.split("_")[1] + "_" + pred_1_or.name.split("_")[2]
            name2 = pred_2_or.name.split("_")[1] + "_" + pred_2_or.name.split("_")[2]
        else:
            name1 = pred_1_or.name.split('_')[0]
            name2 = pred_2_or.name.split('_')[0]
        folder_name = f"{name1}_vs_{name2}"
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        pairwise(pred_1_or, pred_2_or, False, "Union", folder_path, top_k)
        pairwise(pred_1_or, pred_2_or, True, "Union", folder_path, top_k)
        pairwise(pred_1_and, pred_2_and, False, "Intersection", folder_path, top_k)
        pairwise(pred_1_and, pred_2_and, True, "Intersection", folder_path, top_k)


def data_process_for_venn(pred_dict: Dict[str, List[Predictions]], threshold: Optional[float] = 0,
                          target_fpr: Optional[float] = None, option: str = "TPR") -> List[Predictions]:
    """
    Process the data for the Venn diagram for all seeds: get the pred_list
    :param pred_dict: dictionary of Predictions from different attacks, key: attack name, value: list of Predictions of different seeds
    :param threshold: threshold for the comparison (only used when the graph is generated by threshold otherwise None)
    :param target_fpr: target FPR for the comparison (only used when the graph is generated by FPR otherwise None)
    """
    if len(pred_dict) < 2:
        raise ValueError("There is not enough data for comparison.")

    if threshold != 0:
        result_or = []
        result_and = []
        for attack, pred_obj_list in pred_dict.items():
            if option == "TPR":
                common_or, common_and = find_common_tp_pred(pred_obj_list, fpr=target_fpr)
            elif option == "TNR":
                common_or, common_and = find_common_tn_pred(pred_obj_list, fpr=target_fpr)
            result_or.append(common_or)
            result_and.append(common_and)

    elif target_fpr != None:
        result_or = []
        result_and = []

        for attack, adjusted_list in pred_dict.items():
            if option == "TPR":
                common_or, common_and = find_common_tp_pred(adjusted_list, fpr=target_fpr)
            elif option == "TNR":
                common_or, common_and = find_common_tn_pred(adjusted_list, fpr=target_fpr)

            result_or.append(common_or)
            result_and.append(common_and)

    else:
        raise ValueError("Either threshold or target_fpr should be provided.")

    return result_or, result_and

def plot_venn_diagram(pred_or: List[Predictions], pred_and: List[Predictions], save_path: str, signal: bool = False, top_k: bool = False):
    """
    plot venn diagrams based on the goal including both unweighted and weighted venn diagrams for at most 3 attacks.
    :param pred_or: list of Predictions for the 'pred_or' set
    :param pred_and: list of Predictions for the 'pred_and' set
    :param title: title of the graph
    :param save_path: path to save the graph
    :param signal: whether the signal is included in the attack
    :param top_k: whether the top_k is included in the attack
    """
    # if signal, mapped the attack name to the signal name
    attacked_points_or = {pred.name: set() for pred in pred_or}
    attacked_points_and = {pred.name: set() for pred in pred_and}

    venn_sets_or = []
    if signal:
        venn_labels_or = [pred.name.rsplit('_', 1)[0] for pred in pred_or]
    elif top_k:
        venn_labels_or = [pred.name.split("_")[1] + "_" + pred.name.split("_")[2] for pred in pred_or]
        # venn_labels_or = [pred.name.rsplit('_', 1)[0] for pred in pred_or]
    else:
        venn_labels_or = [pred.name.split('_')[0] for pred in pred_or]
    for pred in pred_or:
        attacked_points_or[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets_or.append(attacked_points_or[pred.name])

    venn_sets_and = []
    if signal:
        venn_labels_and = [pred.name.rsplit('_', 1)[0] for pred in pred_and]
    elif top_k:
        venn_labels_and = [pred.name.split("_")[1] + "_" + pred.name.split("_")[2] for pred in pred_and]
        # venn_labels_and = [pred.name.rsplit('_', 1)[0] for pred in pred_and]
    else:
        venn_labels_and = [pred.name.split('_')[0] for pred in pred_and]
    for pred in pred_and:
        attacked_points_and[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets_and.append(attacked_points_and[pred.name])

    total_or = len(set.union(*venn_sets_or))
    total_and = len(set.union(*venn_sets_and))

    # plot venn diagram for the or set
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.axis('off')

    # color mapping
    plt.style.use('seaborn-v0_8-paper')
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    venn_or = venn3(subsets=venn_sets_or, set_labels=venn_labels_or, set_colors=default_colors,
                    subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x / total_or):.1%}" + ")")
    for text in venn_or.set_labels:
        if text:
            text.set_fontsize(23)
            text.set_fontweight('bold')
    for text in venn_or.subset_labels:
        if text:
            text.set_fontsize(16)
            text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "union.pdf"))
    plt.close()

    # plot venn diagram for the and set
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.axis('off')

    plt.style.use('seaborn-v0_8-paper')
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    venn_and = venn3(subsets=venn_sets_and, set_labels=venn_labels_and, set_colors=default_colors,
                     subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x / total_and):.1%}" + ")")
    for text in venn_and.set_labels:
        if text:
            text.set_fontsize(23)
            text.set_fontweight('bold')
    for text in venn_and.subset_labels:
        if text:
            text.set_fontsize(16)
            text.set_fontweight('bold')


    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "intersection.pdf"))
    plt.close()


def plot_venn_for_all_attacks(pred_or: List[Predictions], pred_and: List[Predictions], save_path: str):
    """
    Plot Venn diagrams for at most 5 attacks, supporting both unweighted and weighted diagrams.
    :param pred_or: list of Predictions for the 'pred_or' set
    :param pred_and: list of Predictions for the 'pred_and' set
    :param save_path: path to save the graph
    """
    attacked_points_or = {pred.name: set() for pred in pred_or}
    attacked_points_and = {pred.name: set() for pred in pred_and}

    venn_sets_or = []
    venn_labels_or = [pred.name.split('_')[0] for pred in pred_or]
    for pred in pred_or:
        attacked_points_or[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets_or.append(attacked_points_or[pred.name])

    venn_sets_and = []
    venn_labels_and = [pred.name.split('_')[0] for pred in pred_and]
    for pred in pred_and:
        attacked_points_and[pred.name] = set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0])
        venn_sets_and.append(attacked_points_and[pred.name])

    # calculate the overall jaccard similarity
    jaccard_sim_or = overall_jaccard_similarity(pred_or)
    jaccard_sim_and = overall_jaccard_similarity(pred_and)

    # color mapping
    plt.style.use('seaborn-v0_8-paper')
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    venn_data = [
        (venn_sets_or, venn_labels_or, "Union", jaccard_sim_or, default_colors, "union"),
        (venn_sets_and, venn_labels_and, "Intersection", jaccard_sim_and, default_colors, "intersection")
    ]

    for venn_sets, venn_labels, venn_title, jaccard_sim, default_colors, fname_suffix in venn_data:
        if all(len(s) == 0 for s in venn_sets):
            graph_info = '/'.join(save_path.split('/')[-5:])
            print(f"Skip plotting because all sets are empty. The current path is {graph_info}. "
                  f"We process the original data by finding {venn_title} of all seeds.")
            continue

        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, aspect='equal')
        dataset_dict = {name: data for name, data in zip(venn_labels, venn_sets)}
        venn(dataset_dict, fmt="{size}", cmap=default_colors, fontsize=20, alpha=0.5, legend_loc=None, ax=ax)
        plt.legend(venn_labels, loc='lower left', bbox_to_anchor=(0.75, 0.25), fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{fname_suffix}.pdf"), bbox_inches='tight')
        plt.close()


def compare_models(pred_dict: Dict[str, List[Predictions]], FPR: float, architecture: str, save_path: str):
    """
    Compare the models by listing the number of attacked points of every instance of each attack.
    Save results in both CSV and txt files.
    :param pred_dict: dictionary of Predictions from different attacks.
    :param architecture: architecture of the model.
    :param save_path: path to save the comparison results.
    """
    # Dictionary for matching the attack names
    attack_name_mapping = {
        "aug": "Augmentation attack",
        "losstraj": "Loss trajectory attack",
        "LiRA": "LiRA",
        "reference": "Reference attack",
        "Class-NN": "Class-NN",
        "LOSS": "LOSS",
        "loss-cali": "Difficulty calibration loss attack"
    }

    # check if it is hard or soft label
    for attack, pred_list in pred_dict.items():
        for pred in pred_list:
            if not pred.is_hard():
                pred.pred_arr = pred.adjust_fpr(FPR)

    attacked_points = {}
    for attack, pred_list in pred_dict.items():
        attacked_points[attack] = []
        for pred in pred_list:
            attacked_points[attack].append(set(np.where((pred.pred_arr == 1) & (pred.ground_truth_arr == 1))[0]))

    result = {}
    for attack, attacked_pts in attacked_points.items():
        result[attack] = []
        for i, attacked_pt in enumerate(attacked_pts):
            result[attack].append(len(attacked_pt))
        # find common attacked points
        common_attacked_points = set.intersection(*attacked_pts)
        result[attack].append(len(common_attacked_points))

    # Save results into CSV and .txt file
    csv_file_path = f"{save_path}/{architecture}_attacked_points.csv"
    txt_file_path = f"{save_path}/{architecture}_attacked_points.txt"

    with open(csv_file_path, 'w') as csv_file, open(txt_file_path, 'w') as txt_file:
        # Writing to CSV file
        csv_file.write("Attack,I_1,I_2,I_3,Intersection\n")
        for attack, attacked_pts in result.items():
            csv_file.write(f"{attack},")
            for attacked_pt in attacked_pts:
                csv_file.write(f"{attacked_pt},")
            csv_file.write("\n")

        # Writing to LaTeX-formatted .txt file with matching attack names
        for attack, attacked_pts in result.items():
            attack_name = attack_name_mapping.get(attack,
                                                  attack)  # Match attack names, default to original if not found
            txt_file.write(f"{attack_name} & ")
            txt_file.write(" & ".join(str(attacked_pt) for attacked_pt in attacked_pts))
            if architecture == "wrn32_4":
                txt_file.write(" \\\\")
            txt_file.write("\n")

