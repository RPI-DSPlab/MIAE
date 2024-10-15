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

datasets = ["cifar10", "cifar100"]
attack_list = ["losstraj", "reference", "lira", "calibration"]
seeds = [0, 1, 2, 3, 4, 5]
model = "resnet56"
ensemble_method = "intersection" # "intersection" or "union" or "majority_vote"
output_dir = "/home/data/wangz56/miae_experiment_aug_more_target_data/ensemble_roc"
pred_path = "/home/data/wangz56/miae_experiment_aug_more_target_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

target_datasets = []
for ds in datasets:
    target_datasets.append(TargetDataset.from_dir(ds, f"/home/data/wangz56/miae_experiment_aug_more_target_data/target/{ds}"))

desired_fpr_values = np.logspace(-6, 0, num=5000)  # Adjust 'num' for resolution


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
        multi_attack_ensemble_each_ds[ds] = ensemble_hardpreds(multi_instance_hardpreds[ds], ensemble_method)

    # roc curve
    for ds in target_datasets:
        hard_preds_to_plot = []
        # include the base prediction from each attack
        for attack in attack_list:
            hard_preds_to_plot.append(HardPreds.from_pred(experiment_dict[ds.dataset_name].retrieve_preds(attack, 0), attack, desired_fpr_values))

        # include the multi-attack ensemble with most attacks
        hard_preds_to_plot.append(multi_attack_ensemble_each_ds[ds][max(multi_attack_ensemble_each_ds[ds].keys(), key=lambda x: len(x))])
        # include all ensemble predictions
        # for attack in multi_attack_ensemble_each_ds[ds]:
        #     hard_preds_to_plot.append(multi_attack_ensemble_each_ds[ds][attack])
        plot_roc_hard_preds(hard_preds_to_plot, output_dir + '/' + f"{ds.dataset_name}_{ensemble_method}_ensemble_roc.pdf")
    

if __name__ == "__main__":
    main()
    