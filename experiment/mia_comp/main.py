import utils
import numpy as np


def correct_pred(pred: np.ndarray, attack_set_membership: np.ndarray) -> np.ndarray:
    """element-wise comparison of the prediction and the attack_set_membership, return a boolean array"""
    pred = utils.predictions_to_labels(pred, threshold=0.5)
    return pred == attack_set_membership


def analysis_preds_similarity(correctness_arr1, correctness_arr2, attack1_name, attack2_name):
    """analysis the similarity of the two correctness arrays"""

    # - common correctness: attack1's prediction corrects and attack2's prediction corrects
    common_correctness = np.logical_and(correctness_arr1, correctness_arr2)
    print('------------------------------------------------------------------------------------')
    print(f"common correctness = attack1_correct_pred AND attack2_correct_pred")
    print(f"num of common correctness: {np.sum(common_correctness)} over {len(common_correctness)}")
    print(f"percentage of common correctness: {np.sum(common_correctness) / len(common_correctness):.4f}")
    print('------------------------------------------------------------------------------------')
    # - common incorrectness: attack1's prediction incorrects and attack2's prediction incorrects
    common_incorrectness = np.logical_and(np.logical_not(correctness_arr1), np.logical_not(correctness_arr2))
    print(f"common incorrectness = attack1_incorrect_pred AND attack2_incorrect_pred")
    print(f"num of common incorrectness: {np.sum(common_incorrectness)} over {len(common_incorrectness)}")
    print(f"percentage of common incorrectness: {np.sum(common_incorrectness) / len(common_incorrectness):.4f}")
    print('------------------------------------------------------------------------------------')
    # - attack 1 only correctness: attacks 1 predicted correctly but attack 2 predicted incorrectly
    attack1_only_correctness = np.logical_and(correctness_arr1, np.logical_not(correctness_arr2))
    print(f"{attack2_name}_only_correctness = attack1_correct_pred AND attack2_incorrect_pred")
    print(
        f"num of {attack1_name}_only_correctness: {np.sum(attack1_only_correctness)} over {len(attack1_only_correctness)}")
    print(
        f"percentage of {attack1_name}_only_correctness: {np.sum(attack1_only_correctness) / len(attack1_only_correctness):.4f}")
    print('------------------------------------------------------------------------------------')
    # - attack 2 only correctness: attacks 2 predicted correctly but attack 1 predicted incorrectly
    attack2_only_correctness = np.logical_and(np.logical_not(correctness_arr1), correctness_arr2)
    print(f"{attack2_name}_only_correctness = attack1_incorrect_pred AND attack2_correct_pred")
    print(
        f"num of {attack2_name}_only_correctness: {np.sum(attack2_only_correctness)} over {len(attack2_only_correctness)}")
    print(
        f"percentage of {attack2_name}_only_correctness: {np.sum(attack2_only_correctness) / len(attack2_only_correctness):.4f}")
    print('------------------------------------------------------------------------------------')


if __name__ == '__main__':
    # loading predictions
    pred_shokri = utils.load_predictions("/home/wangz56/comp_mia/preds/cifar10/resnet56/shokri/pred_shokri.npy")
    pred_losstraj = utils.load_predictions("/home/wangz56/comp_mia/preds/cifar10/resnet56/losstraj/pred_losstraj.npy")
    print(f"pearson correlation: {utils.pearson_correlation(pred_shokri, pred_losstraj):.4f}")

    pred_shokri_binary = utils.predictions_to_labels(pred_shokri, threshold=0.5)
    pred_losstraj_binary = utils.predictions_to_labels(pred_losstraj, threshold=0.5)

    # loading the target_dataset
    index_to_data, attack_set_membership = utils.load_target_dataset("/home/wangz56/comp_mia/dataset_save")

    correctness_shokri = correct_pred(pred_shokri_binary, attack_set_membership)
    correctness_losstraj = correct_pred(pred_losstraj_binary, attack_set_membership)

    # analysis the similarity of the two correctness arrays
    analysis_preds_similarity(correctness_shokri, correctness_losstraj, "shokri", "losstraj")

    # obtain different ensemble predictions
    pred_average = utils.averaging_predictions([pred_shokri, pred_losstraj])
    pred_majority_voting = utils.majority_voting([pred_shokri, pred_losstraj])

    # calculate the accuracy
    print(f"\ncorrect rate of shokri: {utils.accuracy(pred_shokri, attack_set_membership):.4f}")
    print(f"correct rate of losstraj: {utils.accuracy(pred_losstraj, attack_set_membership):.4f}")
    print(f"correct rate of average: {utils.accuracy(pred_average, attack_set_membership):.4f}")
    print(f"correct rate of majority_voting: {utils.accuracy(pred_majority_voting, attack_set_membership):.4f}")

    # plot aug_graph
    utils.custom_auc([pred_shokri, pred_losstraj, pred_average, pred_majority_voting], ["shokri", "losstraj", "average", "majority_voting"], attack_set_membership, "AUC graph", "./auc.png")
