
import argparse
import os
import utils
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset


def correct_pred(pred: utils.Predictions) -> np.ndarray:
    """element-wise comparison of the prediction and the attack_set_membership, return a boolean array"""
    return pred.predictions_to_labels() == pred.ground_truth_arr

def correct_pred_with_fpr(pred: utils.Predictions, target_fpr: float) -> np.ndarray:
    """element-wise comparison of the prediction and the attack_set_membership, return a boolean array"""
    return pred.adjust_fpr(target_fpr) == pred.ground_truth_arr

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
        utils.plot_image_by_index(dataset, common_correctness_points, title, path)
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
        utils.plot_image_by_index(dataset, common_incorrectness_points, title, path)
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
        utils.plot_image_by_index(dataset, attack1_only_correctness_points, title, path)
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
        utils.plot_image_by_index(dataset, attack2_only_correctness_points, title, path)
    else:
        print(f"Not enough {attack2_name} only correctness points with label {fixed_label}.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_sample_level_analysis')
    parser.add_argument('--dataset', type=str, default="cifar10", help='the dataset to be used')
    parser.add_argument('--model', type=str, default="vgg16", help='architecture of the model')

    parser.add_argument('--single_action', type=str, default="", help='[venn, auc]')
    args = parser.parse_args()

    # loading predictions
    pred_shokri = utils.load_predictions \
        (f"/data/public/miae_experiment_aug/preds_sd0/{args.dataset}/{args.model}/shokri/pred_shokri.npy")
    pred_losstraj = utils.load_predictions \
        (f"/data/public/miae_experiment_aug/preds_sd0/{args .dataset}/{args.model}/losstraj/pred_losstraj.npy")
    pred_yeom = utils.load_predictions \
        (f"/data/public/miae_experiment_aug/preds_sd0/{args.dataset}/{args.model}/yeom/pred_yeom.npy")
    print(f"pearson correlation: {utils.pearson_correlation(pred_shokri, pred_losstraj):.4f}")

    # loading the target_dataset
    index_to_data, attack_set_membership = utils.load_target_dataset \
        (f"/data/public/miae_experiment_aug/target/{args.dataset}")


    # creating the Predictions object
    pred_shokri_obj = utils.Predictions(pred_shokri, attack_set_membership, "shokri (seed = 0)")
    pred_losstraj_obj = utils.Predictions(pred_losstraj, attack_set_membership, "losstraj (seed = 0)")
    pred_yeom_obj = utils.Predictions(pred_yeom, attack_set_membership, "yeom (seed = 0)")
    # pred_shokri_binary = pred_shokri_obj.predictions_to_labels(threshold=0.5)
    # pred_losstraj_binary = pred_losstraj_obj.predictions_to_labels(threshold=0.5)
    # pred_yeom_binary = pred_yeom_obj.predictions_to_labels(threshold=0.5)

    # loading Iteration Learned
    il_score = utils.load_example_hardness \
        (f"/data/public/example_hardness_aug/{args.dataset}/{args.model}/il/il_score.pkl")

    il = utils.SampleHardness(il_score, "iteration learned")
    il_hist_path = f"./{args.dataset}_{args.model}_il.png"
    il.plot_distribution(il_hist_path)

    il_shokri_tp_path = f"./{args.dataset}_{args.model}_il_shokri_tp.png"
    il.plot_distribution_pred_TP(pred_shokri_obj, il_shokri_tp_path)
    il_loss_traj_tp_path = f"./{args.dataset}_{args.model}_il_losstraj_tp.png"
    il.plot_distribution_pred_TP(pred_losstraj_obj, il_loss_traj_tp_path)
    il_yeom_tp_path = f"./{args.dataset}_{args.model}_il_yeom_tp.png"
    il.plot_distribution_pred_TP(pred_yeom_obj, il_yeom_tp_path)

    correctness_shokri = correct_pred(pred_shokri_obj)
    correctness_losstraj = correct_pred(pred_losstraj_obj)
    # correctness_yeom = correct_pred(pred_yeom_obj)

    # analysis the similarity of the two correctness arrays
    print(f"analysis of the similarity of the two correctness arrays using threshold = 0.5")
    analysis_preds_similarity(correctness_shokri, correctness_losstraj, "shokri", "losstraj")

    # obtain different ensemble predictions
    pred_average = utils.averaging_predictions([pred_shokri_obj, pred_losstraj_obj])
    pred_majority_voting = utils.majority_voting([pred_shokri_obj, pred_losstraj_obj])
    unanimous_voting = utils.unanimous_voting([pred_shokri_obj, pred_losstraj_obj])

    # plot aug_graph
    auc_graph_path = f"./{args.dataset}_{args.model}_auc with seeds.png"
    auc_graph_name = f"{args.dataset} {args.model} auc with seeds"
    # utils.custom_auc([pred_shokri, pred_losstraj, pred_average, pred_majority_voting, unanimous_voting], ["shokri", "losstraj", "average", "majority_voting", "unanimous_voting"], attack_set_membership, auc_graph_name, auc_graph_path)
    utils.custom_auc([pred_shokri, pred_losstraj, pred_yeom], ["shokri", "losstraj", "yeom"], attack_set_membership, auc_graph_name, auc_graph_path)


    pred_average_obj = utils.Predictions(pred_average, attack_set_membership, "average")
    pred_majority_voting_obj = utils.Predictions(pred_majority_voting, attack_set_membership, "majority_voting")
    unanimous_voting_obj = utils.Predictions(unanimous_voting, attack_set_membership, "unanimous_voting")

    # # obtain different seeds object for shokri, the number after the name is the seed
    # pred_shokri_1 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_overfit/preds_sd1/{args.dataset}/{args.model}/shokri/pred_shokri.npy")
    # pred_shokri_2 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_overfit/preds_sd2/{args.dataset}/{args.model}/shokri/pred_shokri.npy")
    # pred_shokri_3 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_overfit/preds_sd3/{args.dataset}/{args.model}/shokri/pred_shokri.npy")
    # pred_shokri_1_obj = utils.Predictions(pred_shokri_1, attack_set_membership, "shokri (seed = 1)")
    # pred_shokri_2_obj = utils.Predictions(pred_shokri_2, attack_set_membership, "shokri (seed = 2)")
    # pred_shokri_3_obj = utils.Predictions(pred_shokri_3, attack_set_membership, "shokri (seed = 3)")
    # pred_shokri_1_binary = pred_shokri_1_obj.predictions_to_labels(threshold=0.5)
    # pred_shokri_2_binary = pred_shokri_2_obj.predictions_to_labels(threshold=0.5)
    # pred_shokri_3_binary = pred_shokri_3_obj.predictions_to_labels(threshold=0.5)
    # pred_shokri_union = np.logical_and(np.logical_and(pred_shokri_binary, pred_shokri_1_binary), pred_shokri_2_binary)
    # pred_shokri_union_obj = utils.Predictions(pred_shokri_union, attack_set_membership, "shokri_union")


    # obtain different seeds object for losstraj, the number after the name is the seed
    # pred_losstraj_1 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_aug/preds_sd1/{args.dataset}/{args.model}/losstraj/pred_losstraj.npy")
    # pred_losstraj_2 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_aug/preds_sd2/{args.dataset}/{args.model}/losstraj/pred_losstraj.npy")
    # pred_losstraj_3 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_aug/preds_sd3/{args.dataset}/{args.model}/losstraj/pred_losstraj.npy")
    # pred_losstraj_1_obj = utils.Predictions(pred_losstraj_1, attack_set_membership, "losstraj (seed = 1)")
    # pred_losstraj_2_obj = utils.Predictions(pred_losstraj_2, attack_set_membership, "losstraj (seed = 2)")
    # pred_losstraj_3_obj = utils.Predictions(pred_losstraj_3, attack_set_membership, "losstraj (seed = 3)")
    # pred_losstraj_1_binary = pred_losstraj_1_obj.predictions_to_labels(threshold=0.5)
    # pred_losstraj_2_binary = pred_losstraj_2_obj.predictions_to_labels(threshold=0.5)
    # pred_losstraj_3_binary = pred_losstraj_3_obj.predictions_to_labels(threshold=0.5)
    # pred_losstraj_union = np.logical_and(np.logical_and(pred_losstraj_binary, pred_losstraj_1_binary), pred_losstraj_2_binary)
    # pred_losstraj_union_obj = utils.Predictions(pred_losstraj_union, attack_set_membership, "losstraj_union")

    # # obtain different seeds object for yeom, the number after the name is the seed
    # pred_yeom_1 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_aug/preds_sd1/{args.dataset}/{args.model}/yeom/pred_yeom.npy")
    # pred_yeom_2 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_aug/preds_sd2/{args.dataset}/{args.model}/yeom/pred_yeom.npy")
    # pred_yeom_3 = utils.load_predictions \
    #     (f"/data/public/miae_experiment_aug/preds_sd3/{args.dataset}/{args.model}/yeom/pred_yeom.npy")
    # pred_yeom_1_obj = utils.Predictions(pred_yeom_1, attack_set_membership, "yeom (seed = 1)")
    # pred_yeom_2_obj = utils.Predictions(pred_yeom_2, attack_set_membership, "yeom (seed = 2)")
    # pred_yeom_3_obj = utils.Predictions(pred_yeom_3, attack_set_membership, "yeom (seed = 3)")
    # pred_yeom_1_binary = pred_yeom_1_obj.predictions_to_labels(threshold=0.5)
    # pred_yeom_2_binary = pred_yeom_2_obj.predictions_to_labels(threshold=0.5)
    # pred_yeom_3_binary = pred_yeom_3_obj.predictions_to_labels(threshold=0.5)
    # pred_yeom_union = np.logical_and(np.logical_and(pred_yeom_binary, pred_yeom_1_binary), pred_yeom_2_binary)
    # pred_yeom_union_obj = utils.Predictions(pred_yeom_union, attack_set_membership, "yeom_union")


    # pred_average3 = utils.averaging_predictions([pred_shokri_3_obj, pred_losstraj_3_obj])
    # pred_majority_voting3 = utils.majority_voting([pred_shokri_3_obj, pred_losstraj_3_obj])
    # unanimous_voting3 = utils.unanimous_voting([pred_shokri_3_obj, pred_losstraj_3_obj])


    # calculate the accuracy
    print(f"\ncorrect rate of shokri: {pred_shokri_obj.accuracy():.4f}")
    print(f"correct rate of losstraj: {pred_losstraj_obj.accuracy():.4f}")
    print(f"correct rate of yeom: {pred_yeom_obj.accuracy():.4f}")
    print(f"correct rate of average: {pred_average_obj.accuracy():.4f}")
    print(f"correct rate of majority_voting: {pred_majority_voting_obj.accuracy():.4f}")
    print(f"correct rate of unanimous_voting: {unanimous_voting_obj.accuracy():.4f}")

    # plot aug_graph
    # auc_graph_path = f"./{args.dataset}_{args.model}_auc with seeds.png"
    # auc_graph_name = f"{args.dataset} {args.model} auc with seeds"
    # # utils.custom_auc([pred_shokri, pred_losstraj, pred_average, pred_majority_voting, unanimous_voting], ["shokri", "losstraj", "average", "majority_voting", "unanimous_voting"], attack_set_membership, auc_graph_name, auc_graph_path)
    # utils.custom_auc([pred_shokri, pred_losstraj, pred_average, pred_majority_voting], ["shokri", "losstraj", "average", "majority_voting"], attack_set_membership, auc_graph_name, auc_graph_path)

    # plot venn diagram for different attacks to compare the similarity
    venn_graph_path_fpr = f"./{args.dataset}_{args.model}_venn_fix_fpr.png"
    venn_graph_name_fpr = f"{args.dataset} {args.model} Venn Diagram for Different Attacks With Fixed FPR = 0.1"
    utils.plot_venn_diagram([pred_shokri_obj, pred_losstraj_obj], venn_graph_name_fpr, venn_graph_path_fpr,
                            goal="different_attacks_fpr", target_fpr=0.1)

    venn_graph_path = f"./{args.dataset}_{args.model}_venn.png"
    venn_graph_name = f"{args.dataset} {args.model} Venn Diagram for Different Attacks"
    utils.plot_venn_diagram([pred_shokri_obj, pred_losstraj_obj], venn_graph_name, venn_graph_path,
                            goal="different_attacks_seed", target_fpr=0)

    # plot the venn diagram for one attack but with different seeds
    # venn_graph_path_shokri_seed = f"./{args.dataset}_{args.model}_venn for shokri with seeds.png"
    # venn_graph_name_shokri_seed = f"{args.dataset} {args.model} Venn Diagram for Shokri with Different Seeds, no aug"
    # utils.plot_venn_diagram([pred_shokri_obj, pred_shokri_1_obj, pred_shokri_2_obj], venn_graph_name_shokri_seed,
    #                         venn_graph_path_shokri_seed, goal="seed_compare")
    #
    # venn_graph_path_losstraj_seed = f"./{args.dataset}_{args.model}_venn for losstraj with seeds.png"
    # venn_graph_name_losstraj_seed = f"{args.dataset} {args.model} Venn Diagram for Losstraj with Different Seeds, no aug"
    # utils.plot_venn_diagram([pred_losstraj_obj, pred_losstraj_1_obj, pred_losstraj_2_obj], venn_graph_name_losstraj_seed,
    #                         venn_graph_path_losstraj_seed, goal="seed_compare")

    # venn_graph_path_yeom_seed = f"./{args.dataset}_{args.model}_venn for yeom with seeds.png"
    # venn_graph_name_yeom_seed = f"{args.dataset} {args.model} venn for yeom with different seeds, with aug"
    # utils.plot_venn_diagram([pred_yeom_obj, pred_yeom_1_obj, pred_yeom_2_obj], venn_graph_name_yeom_seed,
    #                         venn_graph_path_yeom_seed, goal="seed_compare")

    # loading the dataset
    # trainset = utils.load_dataset(f"/data/public/miae_experiment_overfit_aug/target/{args.dataset}/target_trainset.pkl")
    # testset = utils.load_dataset(f"/data/public/miae_experiment_overfit_aug/target/{args.dataset}/target_testset.pkl")
    # fullset = ConcatDataset([trainset, testset])


    # plot tsne graph
    # tsne_graph_path = f"./{args.dataset}_{args.model}_tsne.png"
    # tsne_graph_name = f"{args.dataset} {args.model} tsne"
    # utils.plot_t_sne([pred_shokri_obj, pred_losstraj_obj], fullset, tsne_graph_name, tsne_graph_path)

    # analysis the image
    # analysis_image(fullset, correctness_shokri, correctness_losstraj)
