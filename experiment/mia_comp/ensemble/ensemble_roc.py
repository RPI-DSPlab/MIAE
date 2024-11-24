"""
This script is used to generate ROC curve for the 2 stage ensemble in the paper.
"""

# modify this to set up directory:
DATA_DIR="/home/data/wangz56"

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

mia_name_mapping = {"losstraj": "losstraj", "shokri": "Class-NN", "yeom": "LOSS", "lira": "LIRA", "aug": "aug", "calibration": "calibrated-loss", "reference": "reference"}
mia_color_mapping = {"losstraj": '#1f77b4', "shokri": '#ff7f0e', "yeom": '#2ca02c', "lira": '#d62728', "aug": '#9467bd', "calibration": '#8c564b', "reference": '#e377c2'}




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
        single_instance[attack] = experiment_set.retrieve_preds(attack, 0)
    
    for attack in attack_names:
        all_instances = [experiment_set.retrieve_preds(attack, seed) for seed in seed_list]
        if ensemble_method == "intersection":
            result_pred = np.ones_like(all_instances[0].pred_arr)
            for instance in all_instances:
                result_pred = np.logical_and(result_pred, instance.pred_arr)
            multi_instance[attack] = Predictions(result_pred, single_instance[attack].ground_truth_arr, f"{attack}_intersection")
        elif ensemble_method == "union":
            result_pred = np.zeros_like(all_instances[0].pred_arr)
            for instance in all_instances:
                result_pred = np.logical_or(result_pred, instance.pred_arr)
            multi_instance[attack] = Predictions(result_pred, single_instance[attack].ground_truth_arr, f"{attack}_union")
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
                   log_scale: bool = True, overwrite: bool = False, marker=False, verbose: bool = False):
    """
    This main function would ensemble attacks at some specified FPR values,
    it could be considered as repeating the experiment in the table of the paper for
    different FPRs.

    :param ds: TargetDataset object
    :param log_scale: whether to use log scale for FPR values
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
        attack_perf_df = DataFrame(columns=["Attack", "Ensemble Level", "AUC", "ACC"])
        
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
            
            attack_perf_df = pd.concat([attack_perf_df, DataFrame([{"Attack": attack, "Ensemble Level": ensemble_level, "AUC": auc, "ACC": acc}])], ignore_index=True)
            if verbose:
                print(f"Attack: {attack}, Ensemble Level: {ensemble_level}, AUC: {auc}, ACC: {acc}")
        
        attack_perf_df.to_pickle(path_to_attack_perf_df)
    else:
        roc_df = pd.read_pickle(path_to_roc_df)
        attack_perf_df = pd.read_pickle(path_to_attack_perf_df)
    

    
    # Plotting the data from df with seaborn
    plt.figure()

    # Plot Single Instance
    single_instance_df = roc_df[roc_df['Ensemble Level'] == 'Single Instance']
    sns.lineplot(data=single_instance_df, x='FPR', y='TPR', hue='Attack', style='Ensemble Level', dashes=[(2, 2)], markers=marker, palette=mia_color_mapping, errorbar=None)

    # Plot Multi Instances
    multi_instance_df = roc_df[roc_df['Ensemble Level'] == 'Multi Instances']
    sns.lineplot(data=multi_instance_df, x='FPR', y='TPR', hue='Attack', style='Ensemble Level', dashes=[(1, 0)], markers=marker, palette=mia_color_mapping, errorbar=None)

    # Plot Multi Attacks (ensemble of all attacks)
    multi_attacks_df = roc_df[roc_df['Ensemble Level'] == 'Multi Attacks']
    longest_name = max(multi_attacks_df['Attack'], key=len)
    longest_name_df = multi_attacks_df[multi_attacks_df['Attack'] == longest_name]
    if marker:
        sns.lineplot(data=longest_name_df, x='FPR', y='TPR', color='black', marker='o', label='Multi Attacks (All)')
    else:
        sns.lineplot(data=longest_name_df, x='FPR', y='TPR', color='black', label='Multi Attacks (All)')

    plt.plot([0, 1], [0, 1], ls='--', color='gray')

    if log_scale:
        plt.xlim(1e-4, 1)
        plt.ylim(1e-5, 1)

        plt.xscale('log')
        plt.yscale('log')
    else: 
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

    plt.legend()

    plt.tight_layout()

    # Save the plot
    filename = f"{ds.dataset_name}_{ensemble_method}_roc_plot_liner.pdf" if not log_scale else f"{ds.dataset_name}_{ensemble_method}_roc_plot_log.pdf"
    plt.savefig(save_dir + '/' + filename, format="pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot at {save_dir + '/' + filename}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROC curve for the 2 stage ensemble.")
    parser.add_argument('--datasets', nargs='+', default=["purchase100", "texas100"], help='List of datasets to process.')
    parser.add_argument('--attack_list', nargs='+', default=["losstraj", "reference", "lira", "calibration"], help='List of attacks to process.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help='List of seeds to use.')
    parser.add_argument('--model', type=str, default="mlp_for_texas_purchase", help='Model name.')
    parser.add_argument('--path_to_data', type=str, default=f'{DATA_DIR}/miae_experiment_aug_more_target_data', help='Path to the data directory.')
    parser.add_argument('--num_fpr_for_table_ensemble', type=int, default=100, help='Number of FPR values to ensemble for table.')
    args = parser.parse_args()


    datasets = args.datasets
    attack_list = args.attack_list
    seeds = args.seeds
    model = args.model
    path_to_data = args.path_to_data
    pred_path = path_to_data

    target_datasets = []
    for ds in datasets:
        print(f"Loading from {path_to_data}/target/{ds}")
        target_datasets.append(TargetDataset.from_dir(ds, f"{path_to_data}/target/{ds}"))

    
    # for num_seed in range(2, len(seeds)+1):
    for num_seed in range(2, len(seeds)+1):
        seeds_consider = seeds[:num_seed]
        for ds in target_datasets:
            for ensemble_method in ["intersection", "union"]:
                print(f"Processing {ds.dataset_name} with {num_seed} seeds and ensemble method {ensemble_method}")
                save_dir = f"{path_to_data}/ensemble/{ds.dataset_name}/{num_seed}_seeds/{ensemble_method}"
                os.makedirs(save_dir, exist_ok=True)

                table_roc_main(ds, save_dir, seeds_consider, attack_list, model, args.num_fpr_for_table_ensemble, ensemble_method, log_scale=True, overwrite=False, marker=False)
                table_roc_main(ds, save_dir, seeds_consider, attack_list, model, args.num_fpr_for_table_ensemble, ensemble_method, log_scale=False, overwrite=False, marker=False)

    