"""
This file defines classes/functions for comparing the MIA's predictions down to sample level.
"""
import os
import pickle
import sys


from typing import Optional
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict

sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../../experiment/")
from miae.eval_methods.prediction import Predictions

class TargetDataset():
    """
    A class to store the target dataset and its membership information.
    """
    def __init__(self, name, target_trainset, target_testset, aux_set, index_to_data, membership, dir=None):
        self.dataset_name = name
        self.dir = dir
        self.target_trainset = target_trainset
        self.target_testset = target_testset
        self.aux_set = aux_set
        self.index_to_data = index_to_data
        self.membership = membership

    @classmethod
    def from_dir(cls, name, target_data_dir):
        """alternative constructor to load the target dataset from a directory"""        
        with open(os.path.join(target_data_dir, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(target_data_dir, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(target_data_dir, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)
        index_to_data, membership = load_target_dataset(target_data_dir)
        return cls(name, target_trainset, target_testset, aux_set, index_to_data, membership, dir=target_data_dir)


class ExperimentSet():
    """
    A class to store a set of attack experiment. A set of experiments is defined by the target dataset and
    multi-instances attack predictions on the target dataset.
    """
    def __init__(self, target_dataset: TargetDataset, attack_preds: Dict[str, List[Predictions]], adjusted_fpr=None):
        """
        target_dataset: the target dataset object
        attack_preds: a dictionary mapping attack names to a list of Predictions. 
        adjusted_fpr: the FPR used to adjust the FPR for all preds in the experiment. If None, no adjustment is made.
        """
        self.target_dataset = target_dataset
        self.attack_preds = attack_preds
        self.adjusted_fpr = adjusted_fpr

    @classmethod
    def from_dir(cls, target_dataset: TargetDataset, attack_list: List[str], pred_path: str, sd_list, model, fpr_to_adjust=None):
        """
        alternative constructor to load the experiment set from a directory

        when fpr_to_adjust is not None, the FPR of all predictions in the experiment set will be adjusted to the given FPR.
        if fpr_to_adjust is None, no adjustment is made, predictions are soft (MIA scores)
        """
        attack_preds = dict()
        ret_list = read_preds(pred_path, "", sd_list, target_dataset.dataset_name, model, attack_list, target_dataset.membership)
        for i, a in enumerate(attack_list, 0):
            attack_preds[a] = list()
            for sd in sd_list:
                curr_pred = ret_list[sd][i]
                if not fpr_to_adjust is None:
                    curr_pred = Predictions(curr_pred.adjust_fpr(fpr_to_adjust), curr_pred.ground_truth_arr,
                                            curr_pred.name)
                attack_preds[a].append(curr_pred)
        return cls(target_dataset, attack_preds, fpr_to_adjust)
    
    
    def batch_adjust_fpr(self, fpr_to_adjust):
        """
        adjust the FPR of all predictions in the experiment set
        """
        for attack_name in self.attack_preds.keys():
            for seed in range(len(self.attack_preds[attack_name])):
                self.attack_preds[attack_name][seed] = Predictions(self.attack_preds[attack_name][seed].adjust_fpr(fpr_to_adjust), self.attack_preds[attack_name][seed].ground_truth_arr, self.attack_preds[attack_name][seed].name)
        self.adjusted_fpr = fpr_to_adjust
    
    
    def get_attack_names(self) -> List[str]:
        """
        return the list of attack names in the experiment set
        """
        return list(self.attack_preds.keys())
        
    
    def retrieve_preds(self, attack_name: str, seed: int) -> Predictions:
        """
        retrieve the predictions for a specific attack and seed
        """
        return self.attack_preds[attack_name][seed]

    def get_preds_stability(self, attack_name: str) -> Predictions:
        """
        retrieve the intersection(stability) of the prediction for a specific attack. The stability is the intersection of the
        TP of the predictions of all seeds.
        """
        list_of_base_pred = [self.attack_preds[attack_name][seed].pred_arr for seed in range(len(self.attack_preds[attack_name]))]
        ret_arr = np.array(list_of_base_pred).all(axis=0)
        return Predictions(ret_arr, self.attack_preds[attack_name][0].ground_truth_arr, f"{attack_name}_stability")
    
    def get_preds_coverage(self, attack_name: str) -> Predictions:
        """
        retrieve the union(coverage) of the prediction for a specific attack. The coverage is the union of the
        TP of the predictions of all seeds.
        """
        list_of_base_pred = [self.attack_preds[attack_name][seed].pred_arr for seed in range(len(self.attack_preds[attack_name]))]
        ret_arr = np.array(list_of_base_pred).any(axis=0)
        return Predictions(ret_arr, self.attack_preds[attack_name][0].ground_truth_arr, f"{attack_name}_coverage")
    
    def get_class_unique_TP(self, attack_names: List[str], set_op: str) -> List[Predictions]:
        """
        return the unique TP of the stability of the prediction for a specific attack and seed
        
        :param attack_names: list of attack names
        :param set_op: the set operation to use to combine the unique TP of the attacks. (union or intersection)
        """

        if set_op == "intersection":
            list_of_base_pred_arr = [self.get_preds_stability(a).pred_arr for a in attack_names]
            list_of_base_pred_name = [self.get_preds_stability(a).name for a in attack_names]
        elif set_op == "union":
            list_of_base_pred_arr = [self.get_preds_coverage(a).pred_arr for a in attack_names]
            list_of_base_pred_name = [self.get_preds_coverage(a).name for a in attack_names]
        else:
            raise ValueError(f"set_op must be either 'union' or 'intersection'. Got {set_op} instead.")

        ret_list = []

        for base_pred_idx in range(len(list_of_base_pred_arr)):
            curr_pred = list_of_base_pred_arr[base_pred_idx]
            other_attack_preded = np.zeros_like(list_of_base_pred_arr[base_pred_idx])
            for i in range(len(list_of_base_pred_arr)):
                if i != base_pred_idx:
                    np.logical_or(other_attack_preded, list_of_base_pred_arr[i], out = other_attack_preded)
            curr_pred_unique = np.logical_and(curr_pred, np.logical_not(other_attack_preded))
            ret_list.append(Predictions(curr_pred_unique, self.get_preds_stability(attack_names[base_pred_idx]).ground_truth_arr, list_of_base_pred_name[base_pred_idx] + "_unique_TP"))
        return ret_list
    
    def attack_pair_wise_jaccard_similarity(self, attack_name: str, **kwargs) -> float:
        """
        calculate the pair-wise Jaccard similarity between the predictions of the attack

        Similarity is calculated as:

        Similarity(A) = frac{1}{binom{n}{2}} \sum_{i < j}  J(A_i, A_j)

        For more detail on this, check our paper.

        :param attack_name: the name of the attack

        :return: the Jaccard similarity between the predictions of the attack
        """
        # check if the attack exists, and the predictions should be hard predictions
        if not attack_name in self.attack_preds.keys():
            raise ValueError(f"The attack {attack_name} is not found in the experiment set.")
        for seed in range(len(self.attack_preds[attack_name])):
            if not self.attack_preds[attack_name][seed].is_hard():
                raise ValueError(f"The predictions for {attack_name} is not hard prediction.")
        if seed < 2:
            raise ValueError(f"The attack {attack_name} has less than 2 seeds. Cannot calculate the similarity.")

        if "canary_indices" in kwargs:
            canary_indices = kwargs["canary_indices"] # these are the indices of the canary samples in the trainset

        n = len(self.attack_preds[attack_name])
        total_sim = 0
        for i in range(n):
            for j in range(i+1, n):
                if "canary_indices" in kwargs:
                    total_sim += self.attack_preds[attack_name][i].jaccard_similarity(self.attack_preds[attack_name][j], canary_indices=canary_indices)
                else:
                    total_sim += self.attack_preds[attack_name][i].jaccard_similarity(self.attack_preds[attack_name][j])
        return total_sim / (n * (n - 1) / 2)



def read_pred(preds_path: str, extend_name: str, sd: int, dataset: str, model: str,
              attack: str, gt: np.ndarray) -> Predictions:
    """
    Read the prediction file and return them, the format of prediction follows: f"preds_sd{seed}{extend_name}"

    :param preds_path: the path to the predictions folders
    :param extend_name: the extension of the name of the file (what goes after preds_sd{seed})
    :param sd: seed to obtain the predictions
    :param dataset: the dataset used for the predictions
    :param model: the model used for the predictions
    :param attack: the attack used for the predictions
    :param gt: ground true of the membership prediction
    """

    pred_file = os.path.join(preds_path, f"preds_sd{sd}{extend_name}", dataset, model, attack, f"pred_{attack}.npy")
    pred = np.load(pred_file)
    return Predictions(pred, gt, name=f"{dataset}_{model}_{attack}")


def read_preds(preds_path: str, extend_name: str, sds: List[int], dataset: str, model: str,
               attacks: List[str], gt: np.ndarray) -> List[List[Predictions]]:
    """
    wrapper function to read multiple predictions
    file directory is organized as: seed -> dataset -> model -> attack -> preds_{attack}.pkl

    :param preds_path: the path to the predictions folders
    :param extend_name: the extension of the name of the file (what goes after preds_sd{seed})
    :param sds: list of seed to obtain the predictions
    :param dataset: dataset used for the predictions
    :param model: model used for the predictions
    :param attacks: list of attack used for the predictions
    :param gt: ground true of the membership prediction

    :return: List of predictions
    """

    ret_list = []
    for sd in sds:
        list_x = []
        for attack in attacks:
            list_x.append(read_pred(preds_path, extend_name, sd, dataset, model, attack, gt))
        ret_list.append(list_x)

    return ret_list


def save_accuracy(pred: List[Predictions], file_path: str, target_fpr: Optional[float] = None):
    """
    save the accuracy of the prediction to a .txt file
    :param pred: the prediction object
    :param file: the file to save the accuracy
    """
    try:
        with open(file_path, "w") as file:
            for p in pred:
                file.write(f"{p.name}\n")
                file.write(f"Accuracy (with threshold): {p.accuracy():.4f}\n")
                # Write the accuracy of the prediction using FPR
                file.write(f"Accuracy (with FPR = {target_fpr}): {p.accuracy_at_fpr(target_fpr):.4f}\n\n")
    except IOError as e:
        print(f"Error: Unable to write to file '{file_path}': {e}")
    finally:
        file.close()


def load_predictions(file_path: str) -> np.ndarray:
    """
    Load predictions from a file.

    :param file_path: path to the file containing the predictions
    :return: prediction as a numpy array
    """
    prediction = np.load(file_path)
    return prediction


def load_target_dataset(filepath: str):
    """
    Load the target dataset (the dataset that's used to test the performance of the MIA)

    :param filepath: path to the files ("index_to_data.pkl" and "attack_set_membership.npy")
    :return: index_to_data (dictionary representing the mapping from index to data), attack_set_membership
    """

    # check if those 2 files exist
    if not os.path.exists(filepath + "/index_to_data.pkl") or not os.path.exists(
            filepath + "/attack_set_membership.npy"):
        raise FileNotFoundError(
            f"The files 'index_to_data.pkl' and 'attack_set_membership.npy' are not found in the given path: \n{filepath}.")

    # Load the dataset
    index_to_data = pickle.load(open(filepath + "/index_to_data.pkl", "rb"))
    attack_set_membership = np.load(filepath + "/attack_set_membership.npy")

    return index_to_data, attack_set_membership


def load_dataset(file_path: str) -> Dataset:
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def pearson_correlation(pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    Calculate the Pearson correlation between two predictions.

    :param pred1: prediction 1 as a numpy array
    :param pred2: prediction 2 as a numpy array
    :return: Pearson correlation between the two predictions
    """
    return np.corrcoef(pred1, pred2)[0, 1]


