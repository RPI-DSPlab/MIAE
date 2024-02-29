
import argparse
import os
import utils
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset


def correct_pred(pred: utils.Predictions) -> np.ndarray:
    """element-wise comparison of the prediction and the attack_set_membership, return a boolean array"""
    return pred.predictions_to_labels() == pred.ground_truth_arr


def analysis_preds_similarity(correctness_arr1, correctness_arr2, attack1_name, attack2_name):
    """analysis the similarity of the two correctness arrays"""

    # - common correctness: attack1's prediction corrects and attack2's prediction corrects
    common_correctness = np.logical_and(correctness_arr1, correctness_arr2)
    print('------------------------------------------------------------------------------------')
    print(f"common correctness = {attack1_name}_correct_pred AND {attack2_name}_correct_pred")
    print(f"num of common correctness: {np.sum(common_correctness)} over {len(common_correctness)}")
    print(f"percentage of common correctness: {np.sum(common_correctness) / len(common_correctness):.4f}")
    print('------------------------------------------------------------------------------------')
    # - common incorrectness: attack1's prediction incorrects and attack2's prediction incorrects
    common_incorrectness = np.logical_and(np.logical_not(correctness_arr1), np.logical_not(correctness_arr2))
    print(f"common incorrectness = {attack1_name}_incorrect_pred AND {attack2_name}_incorrect_pred")
    print(f"num of common incorrectness: {np.sum(common_incorrectness)} over {len(common_incorrectness)}")
    print(f"percentage of common incorrectness: {np.sum(common_incorrectness) / len(common_incorrectness):.4f}")
    print('------------------------------------------------------------------------------------')
    # - attack 1 only correctness: attacks 1 predicted correctly but attack 2 predicted incorrectly
    attack1_only_correctness = np.logical_and(correctness_arr1, np.logical_not(correctness_arr2))
    print(f"{attack1_name}_only_correctness = {attack1_name}_correct_pred AND {attack2_name}_incorrect_pred")
    print(
        f"num of {attack1_name}_only_correctness: {np.sum(attack1_only_correctness)} over {len(attack1_only_correctness)}")
    print(
        f"percentage of {attack1_name}_only_correctness: {np.sum(attack1_only_correctness) / len(attack1_only_correctness):.4f}")
    print('------------------------------------------------------------------------------------')
    # - attack 2 only correctness: attacks 2 predicted correctly but attack 1 predicted incorrectly
    attack2_only_correctness = np.logical_and(np.logical_not(correctness_arr1), correctness_arr2)
    print(f"{attack2_name}_only_correctness = {attack1_name}_incorrect_pred AND {attack2_name}_correct_pred")
    print \
        (f"num of {attack2_name}_only_correctness: {np.sum(attack2_only_correctness)} over {len(attack2_only_correctness)}")
    print \
        (f"percentage of {attack2_name}_only_correctness: {np.sum(attack2_only_correctness) / len(attack2_only_correctness):.4f}")
    print('------------------------------------------------------------------------------------')


def analysis_image(dataset: Dataset, correctness_arr1, correctness_arr2):
    """
    load 9 images for each of the following sets
    :param dataset: the dataset to be used
    :param correctness_arr1: the correctness array of attack1
    :param correctness_arr2: the correctness array of attack2
    :param fixed_label: the label to be fixed
    """
    # Basic set up
    attack1_name = "shokri"
    attack2_name = "losstraj"
    fixed_label = 2

    # - common correctness: attack1's prediction corrects and attack2's prediction corrects
    common_correctness = np.logical_and(correctness_arr1, correctness_arr2)
    common_correctness_points = [
        index for index, correct in enumerate(common_correctness) if correct and dataset[index][1] == fixed_label
    ]
    if len(common_correctness_points) >= 9:
        common_correctness_points = np.random.choice(common_correctness_points, 9, replace=False)
        title = f"common correctness with label {fixed_label}"
        path = f"./common_correctness_label_{fixed_label}.png"
        utils.load_image_by_index(dataset, common_correctness_points, title, path)
    else:
        print(f"Not enough common correctness points with label {fixed_label}.")

    # - common incorrectness: attack1's prediction incorrects and attack2's prediction incorrects
    common_incorrectness = np.logical_and(np.logical_not(correctness_arr1), np.logical_not(correctness_arr2))
    common_incorrectness_points = [
        index for index, correct in enumerate(common_incorrectness) if correct and dataset[index][1] == fixed_label
    ]
    if len(common_incorrectness_points) >= 9:
        common_incorrectness_points = np.random.choice(common_incorrectness_points, 9, replace=False)
        title = f"common incorrectness with label {fixed_label}"
        path = f"./common_incorrectness_label_{fixed_label}.png"
        utils.load_image_by_index(dataset, common_incorrectness_points, title, path)
    else:
        print(f"Not enough common incorrectness points with label {fixed_label}.")

    # - attack 1 only correctness: attacks 1 predicted correctly but attack 2 predicted incorrectly
    attack1_only_correctness = np.logical_and(correctness_arr1, np.logical_not(correctness_arr2))
    attack1_only_correctness_points = [
        index for index, correct in enumerate(attack1_only_correctness) if correct and dataset[index][1] == fixed_label
    ]
    if len(attack1_only_correctness_points) >= 9:
        attack1_only_correctness_points = np.random.choice(attack1_only_correctness_points, 9, replace=False)
        title = f"{attack1_name} only correctness with label {fixed_label}"
        path = f"./{attack1_name}_only_correctness_label_{fixed_label}.png"
        utils.load_image_by_index(dataset, attack1_only_correctness_points, title, path)
    else:
        print(f"Not enough {attack1_name} only correctness points with label {fixed_label}.")

    # - attack 2 only correctness: attacks 2 predicted correctly but attack 1 predicted incorrectly
    attack2_only_correctness = np.logical_and(np.logical_not(correctness_arr1), correctness_arr2)
    attack2_only_correctness_points = [
        index for index, correct in enumerate(attack2_only_correctness) if correct and dataset[index][1] == fixed_label
    ]
    if len(attack2_only_correctness_points) >= 9:
        attack2_only_correctness_points = np.random.choice(attack2_only_correctness_points, 9, replace=False)
        title = f"{attack2_name} only correctness with label {fixed_label}"
        path = f"./{attack2_name}_only_correctness_label_{fixed_label}.png"
        utils.load_image_by_index(dataset, attack2_only_correctness_points, title, path)
    else:
        print(f"Not enough {attack2_name} only correctness points with label {fixed_label}.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_prediction')
    parser.add_argument('--dataset', type=str, default="cifar10", help='the dataset to be used')
    parser.add_argument('--model', type=str, default="resnet56", help='architecture of the model')
    args = parser.parse_args()

    # loading the dataset
    trainset = utils.load_dataset(f"/data/public/miae_experiment/dataset_save/{args.dataset}/target_trainset.pkl")
    testset = utils.load_dataset(f"/data/public/miae_experiment/dataset_save/{args.dataset}/target_testset.pkl")
    fullset = ConcatDataset([trainset, testset])

    # loading predictions
    pred_shokri = utils.load_predictions \
        (f"/data/public/miae_experiment/preds/{args .dataset}/{args.model}/shokri/pred_shokri.npy")
    pred_losstraj = utils.load_predictions \
        (f"/data/public/miae_experiment/preds/{args .dataset}/{args.model}/losstraj/pred_losstraj.npy")
    print(f"pearson correlation: {utils.pearson_correlation(pred_shokri, pred_losstraj):.4f}")

    # loading the target_dataset
    index_to_data, attack_set_membership = utils.load_target_dataset \
        (f"/data/public/miae_experiment/dataset_save/{args.dataset}")

    pred_shokri_obj = utils.Predictions(pred_shokri, attack_set_membership, "shokri")
    pred_losstraj_obj = utils.Predictions(pred_losstraj, attack_set_membership, "losstraj")
    pred_shokri_binary = pred_shokri_obj.predictions_to_labels(threshold=0.5)
    pred_losstraj_binary = pred_losstraj_obj.predictions_to_labels(threshold=0.5)

    correctness_shokri = correct_pred(pred_shokri_obj)
    correctness_losstraj = correct_pred(pred_losstraj_obj)

    # analysis the similarity of the two correctness arrays
    analysis_preds_similarity(correctness_shokri, correctness_losstraj, "shokri", "losstraj")

    # obtain different ensemble predictions
    pred_average = utils.averaging_predictions([pred_shokri_obj, pred_losstraj_obj])
    pred_majority_voting = utils.majority_voting([pred_shokri_obj, pred_losstraj_obj])
    unanimous_voting = utils.unanimous_voting([pred_shokri_obj, pred_losstraj_obj])

    pred_average_obj = utils.Predictions(pred_average, attack_set_membership, "average")
    pred_majority_voting_obj = utils.Predictions(pred_majority_voting, attack_set_membership, "majority_voting")
    unanimous_voting_obj = utils.Predictions(unanimous_voting, attack_set_membership, "unanimous_voting")

    # calculate the accuracy
    print(f"\ncorrect rate of shokri: {pred_shokri_obj.accuracy():.4f}")
    print(f"correct rate of losstraj: {pred_losstraj_obj.accuracy():.4f}")
    print(f"correct rate of average: {pred_average_obj.accuracy():.4f}")
    print(f"correct rate of majority_voting: {pred_majority_voting_obj.accuracy():.4f}")
    print(f"correct rate of unanimous_voting: {unanimous_voting_obj.accuracy():.4f}")

    auc_graph_path = f"./{args.dataset}_{args.model}_auc.png"
    auc_graph_name = f"{args.dataset} {args.model} auc"

    # plot aug_graph
    # utils.custom_auc([pred_shokri, pred_losstraj, pred_average, pred_majority_voting, unanimous_voting], ["shokri", "losstraj", "average", "majority_voting", "unanimous_voting"], attack_set_membership, auc_graph_name, auc_graph_path)
    utils.custom_auc([pred_shokri, pred_losstraj, pred_average, pred_majority_voting], ["shokri", "losstraj", "average", "majority_voting"], attack_set_membership, auc_graph_name, auc_graph_path)

    # plot venn diagram
    venn_graph_path = f"./{args.dataset}_{args.model}_venn.png"
    venn_graph_name = f"{args.dataset} {args.model} venn"
    utils.plot_venn_diagram([pred_shokri_obj, pred_losstraj_obj], venn_graph_name, venn_graph_path)

    # plot tsne graph
    tsne_graph_path = f"./{args.dataset}_{args.model}_tsne.png"
    tsne_graph_name = f"{args.dataset} {args.model} tsne"
    utils.plot_t_sne([pred_shokri_obj, pred_losstraj_obj], tsne_graph_name, tsne_graph_path)

    # analysis the image

    analysis_image(fullset, correctness_shokri, correctness_losstraj)