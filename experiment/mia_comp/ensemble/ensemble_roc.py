"""
This script is used to generate ROC curve for the 2 stage ensemble in the paper.
"""

import os
import numpy as np
from typing import List, Tuple, Dict
import itertools

import sys
sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")
from miae.eval_methods.experiment import ExperimentSet, TargetDataset
from miae.eval_methods.prediction import Predictions, HardPreds, plot_roc_hard_preds
from tqdm import tqdm
import argparse

datasets = ["cifar10", "cifar100"]
attack_list = ["losstraj", "reference", "lira", "calibration"]
seeds = [0, 1, 2, 3, 4, 5]
model = "resnet56"
ensemble_method = "intersection" # "intersection" or "union" or "majority_vote"
path_to_data = None
comp_with = None
tp_or_tpr = None

target_datasets = []

desired_fpr_values = np.logspace(-6, 0, num=15000)  # Adjust 'num' for resolution


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



def main():
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROC curve for the 2 stage ensemble.")
    parser.add_argument('--datasets', nargs='+', default=["cifar10", "cifar100"], help='List of datasets to process.')
    parser.add_argument('--attack_list', nargs='+', default=["losstraj", "reference", "lira", "calibration"], help='List of attacks to process.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help='List of seeds to use.')
    parser.add_argument('--model', type=str, default="resnet56", help='Model name.')
    parser.add_argument('--ensemble_method', type=str, default="intersection", choices=["intersection", "union", "majority_vote"], help='Ensemble method to use.')
    parser.add_argument('--path_to_data', type=str, default='/home/data/wangz56/miae_experiment_aug_more_target_data', help='Path to the data directory.')
    parser.add_argument('--comp_with', type=str, default='single_instance', help='compare our ensemble with single instance or multi-instances ensemble.')
    parser.add_argument('--tp_or_tpr', type=str, default='TPR', help='Output directory to save the ROC curve.')
    args = parser.parse_args()


    datasets = args.datasets
    attack_list = args.attack_list
    seeds = args.seeds
    model = args.model
    ensemble_method = args.ensemble_method
    path_to_data = args.path_to_data
    comp_with = args.comp_with
    tp_or_tpr = args.tp_or_tpr

    output_dir = f"{path_to_data}/ensemble_roc" if tp_or_tpr == 'TPR' else f"{path_to_data}/ensemble_roc_tp_count"
    pred_path = path_to_data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_datasets = []
    for ds in datasets:
        target_datasets.append(TargetDataset.from_dir(ds, f"{path_to_data}/target/{ds}"))

    desired_fpr_values = np.logspace(-6, 0, num=15000)  # Adjust 'num' for resolution

    main()
    