"""
This script is used to generate ROC curve for the 2 stage ensemble in the paper.
"""

# modify this to set up directory:
DATA_DIR="data"

import os
import numpy as np
from typing import List, Tuple, Dict
import itertools
from copy import deepcopy
from sklearn.metrics import auc
import pickle


import sys
sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")
from pandas import DataFrame
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import argparse

from miae.eval_methods.experiment import ExperimentSet, TargetDataset
from miae.eval_methods.prediction import Predictions, HardPreds, plot_roc_hard_preds, get_fpr_tpr_hard_label
from experiment.mia_comp.datasets import *


target_datasets = []

desired_fpr_values = np.logspace(-6, 0, num=15000)  # Adjust 'num' for resolution

import matplotlib.pyplot as plt
import matplotlib as mpl

COLUMNWIDTH = 241.14749
COLUMNWIDTH_INCH = 0.01384 * COLUMNWIDTH
TEXTWIDTH = 506.295
TEXTWIDTH_INCH = 0.01384 * TEXTWIDTH

sns.set_context("paper")
# set fontsize
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=7)
plt.rc('font', size=7)       
plt.rc('axes', titlesize=8)    
plt.rc('axes', labelsize=8)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

mia_name_mapping = {"losstraj": "losstraj", "shokri": "Class-NN", "yeom": "LOSS", "lira": "LiRA", "lira_offline": "LiRA (Offline)", "aug": "aug", "calibration": "calibrated-loss", "reference": "reference"}
mia_color_mapping = {"losstraj": '#1f77b4', "shokri": '#ff7f0e', "yeom": '#2ca02c', "lira": '#d62728', "aug": '#9467bd', "calibration": '#8c564b', "reference": '#e377c2', "lira_offline": '#ff7f0e'}

ensemble_name_mapping = {"intersection": "Stability", "union": "Coverage", "majority_vote": "Majority Vote"}



import numpy as np

def interp_closest(x, fprs, tprs):
    """
    Return the TPR from fp corresponding to the FPR in xp that is closest to x,
    along with the closest FPR value found.

    This function mimics the interface of np.interp but instead of linear interpolation,
    it selects the observed value whose FPR is nearest to the target x.

    Parameters
    ----------
    x : float or array_like
        The target FPR value(s) at which to evaluate.
    fprs : 1-D array_like
        The list of FPR values.
    tprs : 1-D array_like
        The list of TPR values corresponding to xp.

    Returns
    -------
    tpr_result, fpr_result : tuple
        - tpr_result: The TPR value(s) corresponding to the FPR closest to x.
        - fpr_result: The closest FPR value(s) found in xp.
    
    Raises
    ------
    ValueError
        If xp and fp do not have the same shape.

    Examples
    --------
    >>> fpr = np.array([0.0, 0.0005, 0.002, 0.01])
    >>> tpr = np.array([0.0, 0.3, 0.7, 1.0])
    >>> tpr_val, fpr_val = interp_closest(0.001, fpr, tpr)
    >>> print(tpr_val)   # 0.3
    >>> print(fpr_val)   # 0.0005
    """
    fprs = np.asarray(fprs)
    tprs = np.asarray(tprs)
    x = np.asarray(x)

    if fprs.shape != tprs.shape:
        raise ValueError("xp and fp must have the same shape.")

    # Handle scalar x
    if x.ndim == 0:
        idx = np.argmin(np.abs(fprs - x))
        return tprs[idx], fprs[idx]
    else:
        distances = np.abs(x[:, None] - fprs[None, :])
        indices = np.argmin(distances, axis=1)
        return tprs[indices], fprs[indices]



def experiment_set_to_hardpreds(experiment_set: ExperimentSet, ensemble_method: str, seed_list: List[int]
                                ) -> Dict[str, HardPreds]:
    """
    Given an experiment set, return a dict of HardPreds for each attack in the ensemble.
    This function is used to generate the first stage of the ensemble.
    """
    
    attack_names = experiment_set.get_attack_names()
    hardpreds_dict = {}
    for attack in tqdm(attack_names, desc="Processing attacks"):
        # retrieve predictions for each seed, then ensemble
        hard_preds_to_ensemble = []
        for seed in seed_list:
            preds = experiment_set.retrieve_preds(attack, seed)
            hard_preds_to_ensemble.append(HardPreds.from_pred(preds, desired_fpr_values))
        hardpreds_dict[attack] = HardPreds.ensemble(hard_preds_to_ensemble, ensemble_method, f"{attack}_{ensemble_method}")

    return hardpreds_dict



def find_combinations_index(num_elements: int):
    """
    Find the list of all combinations of indices, including non-consecutive combinations.
    """
    lst = list(range(num_elements))
    combinations = []
    
    # Use itertools to generate all combinations of all lengths
    for r in range(2, num_elements + 1): 
        combinations.extend(itertools.combinations(lst, r))
    
    return [list(comb) for comb in combinations]



def ensemble_hardpreds(hardpreds_dict: Dict[str, HardPreds], ensemble_method: str) -> Dict[str, HardPreds]:
    """
    ensemble the hardpreds using the ensemble_method for all combinations of attacks.
    This function is used to generate the second stage of the ensemble.
    """

    attack_names = list(hardpreds_dict.keys())
    num_attacks = len(attack_names)
    combinations = find_combinations_index(num_attacks)
    ensemble_hardpreds = {}
    for comb in tqdm(combinations, desc="Ensembling combinations"):
        ensemble_name = "_".join([attack_names[i] for i in comb])
        ensemble_hardpreds[ensemble_name] = HardPreds.ensemble([hardpreds_dict[attack_names[i]] for i in comb], ensemble_method, ensemble_name)

    return ensemble_hardpreds



def ensemble_with_base_FPR(experiment_set: ExperimentSet, ensemble_method: str, seed_list: List[int], base_fpr: float):
    """
    Given an experiment set, carry out multi-instances ensemble with base predictions at base_fpr, then carry out multi-instance ensemble.
    """


    single_instance = {}
    multi_instance = {}
    multi_attacks = {}

    experiment_set = deepcopy(experiment_set)
    experiment_set.batch_adjust_fpr(base_fpr)

    attack_names = experiment_set.get_attack_names()
    for attack in attack_names:
        single_instance[attack] = experiment_set.retrieve_preds(attack, 1)
    
    for attack in attack_names:
        all_instances = [experiment_set.retrieve_preds(attack, seed) for seed in seed_list]
        all_preds = np.stack([instance.pred_arr for instance in all_instances], axis=0)
        if ensemble_method == "intersection":
            result_pred = np.all(all_preds, axis=0)
            multi_instance[attack] = Predictions(result_pred, single_instance[attack].ground_truth_arr, f"{attack}_intersection")
        elif ensemble_method == "union":
            result_pred = np.any(all_preds, axis=0)
            multi_instance[attack] = Predictions(result_pred, single_instance[attack].ground_truth_arr, f"{attack}_union")  
        elif ensemble_method == "majority_vote":
            # Threshold at half the number of instances for majority voting
            result_pred = np.sum(all_preds, axis=0) >= (len(all_instances) / 2)
            multi_instance[attack] = Predictions(result_pred, single_instance[attack].ground_truth_arr, f"{attack}_majority_vote")
        else:
            raise ValueError("Invalid ensemble method.")
        
    
    combination_idx_list = find_combinations_index(len(attack_names))

    for comb in combination_idx_list:
        pred_union = np.zeros_like(single_instance[attack_names[0]].pred_arr)
        for idx in comb:
            pred_union = np.logical_or(pred_union, multi_instance[attack_names[idx]].pred_arr)
        ensemble_name = "_".join([attack_names[i] for i in comb]) + f"_{ensemble_method}"
        multi_attacks[ensemble_name] = Predictions(pred_union, single_instance[attack_names[0]].ground_truth_arr, ensemble_name)

    return single_instance, multi_instance, multi_attacks


def sweep(fpr, tpr) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Modified sweep adapted from 
    https://github.com/tensorflow/privacy/blob/637f17ea4e8326cba1ea4a2ca76fef14b14e51db/research/mi_lira_2021/plot.py
    This is used for the paper Membership Inference Attacks From First Principles by Carlini et al.
    It finds the best balance accuracy at each FPR.
    """
    zip(*sorted(zip(fpr, tpr)))
    fpr, tpr = np.array(fpr), np.array(tpr)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc
    
    

def table_roc_main(ds, save_dir, seeds, attack_list, model, num_fpr_for_table_ensemble, ensemble_method,
                   log_scale: bool = True, overwrite: bool = False, marker=False, verbose: bool = False,
                   additional_command: List[str] = None, no_interp = False
                   ):
    """
    This main function would ensemble attacks at some specified FPR values,
    it could be considered as repeating the experiment in the table of the paper for
    different FPRs.

    :param ds: TargetDataset object
    
    """

    fprs_for_base = np.logspace(-6, 0, num=num_fpr_for_table_ensemble) if log_scale else np.linspace(0, 1, num=num_fpr_for_table_ensemble)
    experiment_dict = {}

    experiment_dict[ds.dataset_name] = ExperimentSet.from_dir(ds, attack_list, pred_path, seeds, model)

    gt = experiment_dict[ds.dataset_name].retrieve_preds(attack_list[0], 0).ground_truth_arr

    path_to_roc_df = f"{save_dir}/ensemble_tpr_fpr.pkl"
    path_to_attack_perf_df = f"{save_dir}/ensemble_perf.pkl"

    if (not os.path.exists(path_to_roc_df) or not os.path.exists(path_to_attack_perf_df)) or overwrite:
        os.makedirs(save_dir + '/' + f"{ds.dataset_name}/{ensemble_method}", exist_ok=True)
        # dataframe to store the results
        """Attack: str, Ensemble Level: str, FPR: float, TPR: float"""
        roc_df = DataFrame(columns=["Attack", "Ensemble Level", "FPR", "TPR"])

        for fpr in fprs_for_base:
            single_instance, multi_instance, multi_attacks = ensemble_with_base_FPR(experiment_dict[ds.dataset_name], ensemble_method, seeds, fpr)
            for (name, pred) in single_instance.items():
                calc_fpr, calc_tpr = get_fpr_tpr_hard_label(pred.pred_arr, gt)
                roc_df = pd.concat([roc_df, DataFrame([{"Attack": name, "Ensemble Level": "Single Instance", "FPR": calc_fpr, "TPR": calc_tpr}])], ignore_index=True)
            for (name, pred) in multi_instance.items():
                calc_fpr, calc_tpr = get_fpr_tpr_hard_label(pred.pred_arr, gt)
                roc_df = pd.concat([roc_df, DataFrame([{"Attack": name, "Ensemble Level": "Multi Instances", "FPR": calc_fpr, "TPR": calc_tpr}])], ignore_index=True)
            for (name, pred) in multi_attacks.items():
                fpr, calc_tpr = get_fpr_tpr_hard_label(pred.pred_arr, gt)
                roc_df = pd.concat([roc_df, DataFrame([{"Attack": name, "Ensemble Level": "Multi Attacks", "FPR": calc_fpr, "TPR": calc_tpr}])], ignore_index=True)

        roc_df.to_pickle(path_to_roc_df)

        # Another dataframe to store auc
        attack_perf_df = DataFrame(columns=["Attack", "Ensemble Level", "AUC", "ACC", "TPR@0.001FPR", "FPR"])
        
        # find all pairs of (attack, ensemble_level) in tpr_fpr_df
        attack_ensemble_set = set([(row["Attack"], row["Ensemble Level"]) for _, row in roc_df.iterrows()])

        # retrieve the TPR and FPR values for each pair for calculating AUC
        for (attack, ensemble_level) in attack_ensemble_set:
            filtered_df = roc_df[(roc_df["Attack"] == attack) & (roc_df["Ensemble Level"] == ensemble_level)]
            fpr = []
            tpr = []
            for entry in filtered_df.iterrows():
                fpr.append(entry[1]["FPR"])
                tpr.append(entry[1]["TPR"])
            
            _, _, auc, acc = sweep(fpr, tpr)
            if no_interp == False:
                fpr_to_save = 0.001
                TPRat0001FPR = np.interp(0.001, fpr, tpr)

            if no_interp: # in this  case, we want actual datapoint instead of interpolated
                TPRat0001FPR, fpr_to_save = interp_closest(0.001, fpr, tpr)
            
            attack_perf_df = pd.concat([attack_perf_df, DataFrame([{"Attack": attack, "Ensemble Level": ensemble_level, "AUC": auc, 
                                                                    "ACC": acc, "TPR@0.001FPR": TPRat0001FPR, "FPR": fpr_to_save}])], ignore_index=True)
            if verbose:
                print(f"Attack: {attack}, Ensemble Level: {ensemble_level}, AUC: {auc}, ACC: {acc}, TPR@0.001FPR: {TPRat0001FPR}")
        
        attack_perf_df.to_pickle(path_to_attack_perf_df)
    else:
        roc_df = pd.read_pickle(path_to_roc_df)
        attack_perf_df = pd.read_pickle(path_to_attack_perf_df)
    

    
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.set_context("paper")

    # Plot Single Instance
    if "don't show single instance" not in additional_command:
        single_instance_df = roc_df[roc_df['Ensemble Level'] == 'Single Instance']
        sns.lineplot(
            ax=ax,
            data=single_instance_df,
            x='FPR',
            y='TPR',
            hue='Attack',
            style='Ensemble Level',
            dashes=[(2, 2)],
            markers=marker,
            palette=mia_color_mapping
        )

    # Plot Multi Instances
    if "don't show multi instance" not in additional_command:
        multi_instance_df = roc_df[roc_df['Ensemble Level'] == 'Multi Instances']
        sns.lineplot(
            ax=ax,
            data=multi_instance_df,
            x='FPR',
            y='TPR',
            hue='Attack',
            style='Ensemble Level',
            dashes=[(1, 0)],
            markers=marker,
            palette=mia_color_mapping
        )

    # Plot Multi Attacks (ensemble of all attacks)
    multi_attacks_df = roc_df[roc_df['Ensemble Level'] == 'Multi Attacks']
    longest_name = max(multi_attacks_df['Attack'], key=len)
    if "show all multi attack" not in additional_command:
        multi_attacks_df = multi_attacks_df[multi_attacks_df['Attack'] == longest_name]
        sns.lineplot(
            ax=ax,
            data=multi_attacks_df,
            x='FPR',
            y='TPR',
            color='black',
            marker='o' if marker else None,
            label='Multi Attacks (All)'
        )
    else:
        sns.lineplot(
            ax=ax,
            data=multi_attacks_df,
            x='FPR',
            y='TPR',
            hue='Attack',
            style='Ensemble Level',
            markers=marker if marker else False
        )

    ax.plot([0, 1], [0, 1], ls='--', color='gray')

    if log_scale:
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    handles, labels = ax.get_legend_handles_labels()
    current_legend = ax.get_legend()
    if current_legend is not None:
        current_legend.remove()

    filename = f"{ds.dataset_name}_{ensemble_method}_roc_plot_linear.pdf" if not log_scale else f"{ds.dataset_name}_{ensemble_method}_roc_plot_log.pdf"
    plt.savefig(save_dir + '/' + filename, format="pdf", bbox_inches='tight')
    print(f"Saved plot at {save_dir + '/' + filename}")
    plt.close(fig)

    # Create separate legend figure with all handles
    legend_fig, legend_ax = plt.subplots(figsize=(3, 2))
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center', frameon=False, ncol=1)
    legend_filename = f"{ds.dataset_name}_{ensemble_method}_roc_legend_linear.pdf" if not log_scale else f"{ds.dataset_name}_{ensemble_method}_roc_legend_log.pdf"
    legend_fig.savefig(save_dir + '/' + legend_filename, format="pdf", bbox_inches='tight')
    plt.close(legend_fig)
    print(f"Saved legend at {save_dir + '/' + legend_filename}")

def compare_ensemble_curves(ds, save_dir, seeds, attack_list, model, num_fpr_for_table_ensemble, ensemble_methods,
                   log_scale: bool = True, overwrite: bool = False, marker=False, verbose: bool = False,
                   additional_command: List[str] = None
                     ):              
                   
    """
    Compare the ROC curves of different ensemble methods (with all 4 attacks) for the same dataset.

    Make sure for selected ensemble methods, the ROC curves are already generated.
    """

    df = pd.DataFrame(columns=["Attack", "FPR", "TPR", "Ensemble Method"])

    # iterate through all ensemble methods and add to the dataframe for only all attacks
    for ensemble_method in ensemble_methods:
        path_to_roc_df = f"{save_dir}/{model}/{ds.dataset_name}/{len(seeds)}_seeds/{ensemble_method}/ensemble_tpr_fpr.pkl"
        roc_df = pd.read_pickle(path_to_roc_df)
        # find longest name
        longest_name = max(roc_df['Attack'], key=len)
        multi_attacks_df = roc_df[roc_df['Attack'] == longest_name]
        # adding the ensemble method to the dataframe
        multi_attacks_df.loc[:, 'Ensemble Method'] = ensemble_name_mapping[ensemble_method]
        df = pd.concat([df, multi_attacks_df])

    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.set_context("paper")


    # Plot Multi Attacks (ensemble of all attacks)
    longest_name = max(df['Attack'], key=len)
    sns.lineplot(
        ax=ax,
        data=df,
        x='FPR',
        y='TPR',
        hue='Ensemble Method'
    )

    ax.plot([0, 1], [0, 1], ls='--', color='gray')

    if log_scale:
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    filename = f"{ds.dataset_name}_ensemble_comparison_roc_plot_linear.pdf" if not log_scale else f"{ds.dataset_name}_ensemble_comparison_roc_plot_log.pdf"
    plt.savefig(save_dir + '/' + filename, format="pdf", bbox_inches='tight')
    print(f"Saved plot at {save_dir + '/' + filename}")
    plt.close(fig)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROC curve for the 2 stage ensemble.")
    parser.add_argument('--datasets', nargs='+', default=["cifar100", "cinic10", "cifar10"], help='List of datasets to process.')
    parser.add_argument('--attack_list', nargs='+', default=["losstraj", "reference", "lira", "calibration"], help='List of attacks to process.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help='List of seeds to use.')
    parser.add_argument('--models', nargs='+', type=str, default="vgg16", help='Model name.')
    parser.add_argument('--path_to_data', type=str, default=f'{DATA_DIR}/miae_experiment_aug_more_target_data', help='Path to the data directory.')
    parser.add_argument('--num_fpr_for_table_ensemble', type=int, default=100, help='Number of FPR values to ensemble for table.')
    parser.add_argument('--no_interp', type=str, help='Use actual data point instead of interpolated value.')
    args = parser.parse_args()

    # additional_command = ["show all multi attack", "don't show single instance", "don't show multi instance"]
    additional_command = []


    datasets = args.datasets
    attack_list = args.attack_list
    seeds = args.seeds
    models = args.models
    path_to_data = args.path_to_data
    pred_path = path_to_data
    no_interp = bool(args.no_interp)

    target_datasets = []
    for ds in datasets:
        print(f"Loading from {path_to_data}/target/{ds}")
        target_datasets.append(TargetDataset.from_dir(ds, f"{path_to_data}/target/{ds}"))

    
    for model in models:
        for num_seed in range(2, len(seeds)+1):
        # for num_seed in [6]:
            seeds_consider = seeds[:num_seed]
            for ds in target_datasets:
                for ensemble_method in ["intersection", "union", "majority_vote"]:
                    print(f"Processing {ds.dataset_name} with {num_seed} seeds and ensemble method {ensemble_method} and model {model}")
                    save_dir = f"{path_to_data}/ensemble_roc_base_sd_1/{model}/{ds.dataset_name}/{num_seed}_seeds/{ensemble_method}"
                    # save_dir = f"{path_to_data}/ensemble_roc/{model}/{ds.dataset_name}/{num_seed}_seeds/{ensemble_method}"
                    os.makedirs(save_dir, exist_ok=True)

                    table_roc_main(ds, save_dir, seeds_consider, attack_list, model, args.num_fpr_for_table_ensemble, ensemble_method, log_scale=True, overwrite=False, marker=False, additional_command=additional_command, no_interp=no_interp)
                    table_roc_main(ds, save_dir, seeds_consider, attack_list, model, args.num_fpr_for_table_ensemble, ensemble_method, log_scale=False, overwrite=False, marker=False, additional_command=additional_command, no_interp=no_interp)

                # # compare the ensemble methods
                # save_dir = f"{path_to_data}/ensemble_roc_base_sd_1/"
                # ensemble_methods = ["intersection", "union", "majority_vote"]
                # compare_ensemble_curves(ds, save_dir, seeds_consider, attack_list, model, args.num_fpr_for_table_ensemble, ensemble_methods, log_scale=True, overwrite=False, marker=False, additional_command=additional_command)
                # compare_ensemble_curves(ds, save_dir, seeds_consider, attack_list, model, args.num_fpr_for_table_ensemble, ensemble_methods, log_scale=False, overwrite=False, marker=False, additional_command=additional_command)

        