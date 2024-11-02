"""
This script is used to generate ROC curve for the 2 stage ensemble in the paper.
"""

import os
import numpy as np
from typing import List, Tuple, Dict
import itertools
from copy import deepcopy
from matplotlib.lines import Line2D


import sys
sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")
from miae.eval_methods.experiment import ExperimentSet, TargetDataset
from miae.eval_methods.prediction import Predictions, HardPreds, plot_roc_hard_preds, get_fpr_tpr_hard_label
from pandas import DataFrame
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import argparse

datasets = ["cifar10", "cifar100"]
attack_list = ["losstraj", "reference", "lira", "calibration"]
seeds = [0, 1, 2, 3, 4, 5]
model = "resnet56"
ensemble_method = "intersection" # "intersection" or "union" or "majority_vote"
pred_path = None
path_to_data = None
comp_with = None
tp_or_tpr = None
seed_to_retrieve = None
num_fpr_for_table_ensemble = None

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



def smooth_roc_main(args):
    """
    This main function would plot smooth ROC plots of the
    ensemble predictions for logspace sampled FPR values.
    """
    experiment_dict = {}
    for ds in target_datasets:
        experiment_dict[ds.dataset_name] = ExperimentSet.from_dir(ds, attack_list, pred_path, seeds, model)

    gt = experiment_dict[datasets[0]].retrieve_preds(attack_list[0], 0).ground_truth_arr
    
    # constructing HardPreds and ensemble for Multi-instances ensemble step
    multi_instance_hardpreds = {}
    for ds in target_datasets:
        multi_instance_hardpreds[ds] = experiment_set_to_hardpreds(experiment_dict[ds.dataset_name], ensemble_method, seeds)

    
    # multi-attack ensemble
    multi_attack_ensemble_each_ds = {}
    for ds in target_datasets:
        multi_attack_ensemble_each_ds[ds] = ensemble_hardpreds(multi_instance_hardpreds[ds], "intersection")

    # roc curve
    for ds in target_datasets:
        hard_preds_to_plot = []
        # include the base prediction from each attack
        if comp_with == 'single_instance':
            for attack in attack_list:
                hard_preds_to_plot.append(HardPreds.from_pred(experiment_dict[ds.dataset_name].retrieve_preds(attack, 1), attack, desired_fpr_values))
        elif comp_with == 'multi_instance':
            for attack in multi_instance_hardpreds[ds]:
                hard_preds_to_plot.append(multi_instance_hardpreds[ds][attack])
        else:
            raise ValueError("comp_with should be either 'single_instance' or 'multi_instance'")
        
        # include the multi-attack ensemble with most attacks
        hard_preds_to_plot.append(multi_attack_ensemble_each_ds[ds][max(multi_attack_ensemble_each_ds[ds].keys(), key=lambda x: len(x))])
        # include all ensemble predictions
        # for attack in multi_attack_ensemble_each_ds[ds]:
        #     hard_preds_to_plot.append(multi_attack_ensemble_each_ds[ds][attack])
        plot_roc_hard_preds(hard_preds_to_plot, output_dir + '/' + f"{ds.dataset_name}_{ensemble_method}_{comp_with}_ensemble_roc.pdf", tp_or_tpr)


"""Below are the functions for the alternative main to match observation in the ensemble result presented
in the paper body."""

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

    
    

def table_roc_main(args):
    """
    This main function would ensemble attacks at some specified FPR values,
    it could be considered as repeating the experiment in the table of the paper for
    different FPRs.
    """

    fprs_for_base = np.logspace(-6, 0, num=args.num_fpr_for_table_ensemble)
    experiment_dict = {}
    for ds in target_datasets:
        experiment_dict[ds.dataset_name] = ExperimentSet.from_dir(ds, args.attack_list, pred_path, args.seeds, args.model)

        gt = experiment_dict[ds.dataset_name].retrieve_preds(args.attack_list[0], 0).ground_truth_arr

        # dataframe to store the results
        """Attack: str, Ensemble Level: str, FPR: float, TPR: float"""
        df = DataFrame(columns=["Attack", "Ensemble Level", "FPR", "TPR"])

        for fpr in fprs_for_base:
            single_instance, multi_instance, multi_attacks = ensemble_with_base_FPR(experiment_dict[ds.dataset_name], args.ensemble_method, args.seeds, fpr)
            for (name, pred) in single_instance.items():
                calc_fpr, calc_tpr = get_fpr_tpr_hard_label(pred.pred_arr, gt)
                df = pd.concat([df, DataFrame([{"Attack": name, "Ensemble Level": "Single Instance", "FPR": calc_fpr, "TPR": calc_tpr}])], ignore_index=True)
            for (name, pred) in multi_instance.items():
                calc_fpr, calc_tpr = get_fpr_tpr_hard_label(pred.pred_arr, gt)
                df = pd.concat([df, DataFrame([{"Attack": name, "Ensemble Level": "Multi Instances", "FPR": calc_fpr, "TPR": calc_tpr}])], ignore_index=True)
            for (name, pred) in multi_attacks.items():
                fpr, calc_tpr = get_fpr_tpr_hard_label(pred.pred_arr, gt)
                df = pd.concat([df, DataFrame([{"Attack": name, "Ensemble Level": "Multi Attacks", "FPR": calc_fpr, "TPR": calc_tpr}])], ignore_index=True)


        df.to_csv(output_dir + '/' + f"{ds.dataset_name}_{ensemble_method}_ensemble_roc_table.csv", index=False)
        

        
        # Plotting the data from df with seaborn
        plt.figure()

        # Plot Single Instance
        single_instance_df = df[df['Ensemble Level'] == 'Single Instance']
        sns.lineplot(data=single_instance_df, x='FPR', y='TPR', hue='Attack', style='Ensemble Level', dashes=[(2, 2)], palette=mia_color_mapping, errorbar=None)

        # Plot Multi Instances
        multi_instance_df = df[df['Ensemble Level'] == 'Multi Instances']
        sns.lineplot(data=multi_instance_df, x='FPR', y='TPR', hue='Attack', style='Ensemble Level', dashes=[(1, 0)], palette=mia_color_mapping, errorbar=None)

        # Plot Multi Attacks (ensemble of all attacks)
        multi_attacks_df = df[df['Ensemble Level'] == 'Multi Attacks']
        longest_name = max(multi_attacks_df['Attack'], key=len)
        longest_name_df = multi_attacks_df[multi_attacks_df['Attack'] == longest_name]
        sns.lineplot(data=longest_name_df, x='FPR', y='TPR', color='black', label='Multi Attacks (All)')

        plt.plot([0, 1], [0, 1], ls='--', color='gray')

        plt.xlim(1e-4, 1)
        plt.ylim(1e-5, 1)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')

        plt.legend()

        plt.tight_layout()

        # Save the plot
        plt.savefig(output_dir + '/' + f"{ds.dataset_name}_{ensemble_method}_{comp_with}_ensemble_roc_plot.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        print(f"Saved plot at {output_dir + '/' + f'{ds.dataset_name}_{ensemble_method}_{comp_with}_ensemble_roc_plot.pdf'}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROC curve for the 2 stage ensemble.")
    parser.add_argument('--mode', type=str, default="table", choices=["smooth", "table"], help='Mode to run the script.')
    parser.add_argument('--datasets', nargs='+', default=["cifar10", "cifar100"], help='List of datasets to process.')
    parser.add_argument('--attack_list', nargs='+', default=["losstraj", "reference", "lira", "calibration"], help='List of attacks to process.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help='List of seeds to use.')
    parser.add_argument('--model', type=str, default="resnet56", help='Model name.')
    parser.add_argument('--ensemble_method', type=str, default="intersection", choices=["intersection", "union", "majority_vote"], help='Ensemble method to use.')
    parser.add_argument('--path_to_data', type=str, default='/home/data/wangz56/miae_experiment_aug_more_target_data', help='Path to the data directory.')
    parser.add_argument('--comp_with', type=str, default='single_instance', help='compare our ensemble with single instance or multi-instances ensemble.')
    parser.add_argument('--tp_or_tpr', type=str, default='TPR', help='Output directory to save the ROC curve.')
    parser.add_argument('--seed_to_retrieve', type=int, default=0, help='Seed to retrieve the predictions for comparison.')
    parser.add_argument('--num_fpr_for_table_ensemble', type=int, default=30, help='Number of FPR values to ensemble for table.')
    args = parser.parse_args()


    # -- update the global variables for plotting --
    datasets = args.datasets
    attack_list = args.attack_list
    seeds = args.seeds
    model = args.model
    ensemble_method = args.ensemble_method
    path_to_data = args.path_to_data
    comp_with = args.comp_with
    tp_or_tpr = args.tp_or_tpr
    seed_to_retrieve = args.seed_to_retrieve

    if args.mode == "smooth":
        output_dir = f"{path_to_data}/ensemble_roc_smooth" if tp_or_tpr == 'TPR' else f"{path_to_data}/ensemble_roc_smooth_tp_count"
    else:
        output_dir = f"{path_to_data}/ensemble_roc_table"

    pred_path = path_to_data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_datasets = []
    for ds in datasets:
        target_datasets.append(TargetDataset.from_dir(ds, f"{path_to_data}/target/{ds}"))

    desired_fpr_values = np.logspace(-6, 0, num=15000)  # Adjust 'num' for resolution

    if args.mode == "smooth":
        main = smooth_roc_main
    elif args.mode == "table":
        main = table_roc_main
    else:
        raise ValueError("Invalid mode, should be either 'smooth' or 'table'.")
    
    main(args)