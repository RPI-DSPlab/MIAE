import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from experiment.llm_models import get_model
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import load_mimir_dataset
from miae.attacks.base import LLM_ModelAccess
from miae.attacks_on_llm.mink import MinKProbAttack, MinKAuxiliaryInfo
from miae.attacks_on_llm.loss import LossAttack, LossAttackAuxiliaryInfo



current_dir = os.getcwd()
attack_dir = os.path.join(current_dir, "attack")
savedir = os.path.join(current_dir, "results")
seed = 0


def plot_auc_roc(attack, member_scores, non_member_scores, save_path):
    """
    Plot the AUC-ROC curve for the attack
    :param attack: The attack we are plotting the curve for
    :param member_scores: The scores for members
    :param non_member_scores: The scores for non-members
    :param output_path: The path to
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scores = np.concatenate([member_scores, non_member_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])

    auc_score = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curve for {attack}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    return auc_score

def plot_distribution(attack, member_scores, non_member_scores, save_path):
    """
    Plot the distribution of the attack for members and non-members
    :param attack: The attack we are plotting the distribution for
    :param member_scores: The scores for members
    :param non_member_scores: The scores for non-members
    :param output_path: The path to save the plot
    """
    plt.hist(member_scores, bins=50, alpha=0.5, label="Members")
    plt.hist(non_member_scores, bins=50, alpha=0.5, label="Non-Members")
    plt.xlabel(f"{attack} Scores")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Distribution of {attack} Scores")
    plt.savefig(save_path)
    plt.close()


def run_minK_attack(attack_config, target_model, train_set, test_set, device):
    """
    Run the MinK attack on the target model
    :param attack_config: The configuration for the attack
    :param target_model: The target model to attack
    :param train_set: The training set to use
    :param test_set: The test set to use
    :param device: The device to use
    """
    mink_info = MinKAuxiliaryInfo(attack_config)
    min_info_dict = mink_info.save_config_to_dict()
    mink_attack = MinKProbAttack(target_model, min_info_dict)

    member_scores = []
    labels = []
    non_member_scores = []

    # Use tqdm to display progress for the train set
    for idx, document in enumerate(tqdm(train_set, desc="Processing Train Set")):
        log_probs_data = mink_attack.target_model.get_signal_llm(
            text=document['text'],
            no_grads=True,
            return_all_probs=True
        )
        probs = torch.tensor(log_probs_data['all_token_log_probs'], device=device)
        score = mink_attack._attack(document=document['text'], probs=probs)
        member_scores.append(score)
        labels.append(1)


    # Use tqdm to display progress for the test set
    for idx, document in enumerate(tqdm(test_set, desc="Processing Test Set")):
        log_probs_data = mink_attack.target_model.get_signal_llm(
            text=document['text'],
            no_grads=True,
            return_all_probs=True
        )
        probs = torch.tensor(log_probs_data['all_token_log_probs'], device=device)
        score = mink_attack._attack(document=document['text'], probs=probs)
        non_member_scores.append(score)

    best_threshold = mink_attack.find_optimal_threshold(member_scores, labels)
    member_classifications = [1 if score < best_threshold else 0 for score in member_scores]
    non_member_classifications = [0 if score < best_threshold else 1 for score in non_member_scores]

    # Plot the AUC-ROC curve
    print("Ready to plot AUC ROC")
    save_path = "/data/public/llm_mia_comp/demo_results/graphs/auc_roc_mink.png"
    auc_score = plot_auc_roc("Min K", member_scores, non_member_scores, save_path)

    print("MinK Attack Results:")
    print(f"AUC-ROC Score: {auc_score}")

def run_loss_attack(attack_config, target_model, train_set, test_set):
    loss_info = LossAttackAuxiliaryInfo(attack_config)
    config_dict = loss_info.save_config_to_dict()
    loss_attack = LossAttack(target_model, config_dict)

    member_scores = []
    non_member_scores = []

    # Use tqdm to display progress for the train set
    for idx, document in enumerate(tqdm(train_set, desc="Processing Train Set for LOSS")):
        tokens = loss_attack.target_model.tokenizer.encode(document['text'])
        score = loss_attack._attack(document=document['text'], tokens=tokens)
        member_scores.append(score)

    # Use tqdm to display progress for the test set
    for idx, document in enumerate(tqdm(test_set, desc="Processing Test Set for LOSS")):
        tokens = loss_attack.target_model.tokenizer.encode(document['text'])
        score = loss_attack._attack(document=document['text'], tokens=tokens)
        non_member_scores.append(score)

    threshold = attack_config['threshold']
    member_classifications = [1 if score < threshold else 0 for score in member_scores]
    non_member_classifications = [0 if score < threshold else 1 for score in non_member_scores]

    # Plot the AUC-ROC curve
    save_path = "/data/public/llm_mia_comp/demo_results/graphs/auc_roc_loss.png"
    auc_score = plot_auc_roc("LOSS", member_scores, non_member_scores, save_path)

    print("Loss Attack Results:")
    print(f"AUC-ROC Score: {auc_score}")


def main():
    # Set up
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model_name = 'pythia_160m'
    model, tokenizer = get_model(model_name, device)
    target_model = LLM_ModelAccess(model, tokenizer, device)

    # Load the dataset
    cache_dir = "/data/public/huggingface_datasets"
    split_dataset = load_mimir_dataset("arxiv", "ngram_13_0.8", cache_dir, test_size=0.5, seed=1)
    train = split_dataset['train']
    test = split_dataset['test']

    # MinK Attack
    print("Running the MinK attack")
    mink_config = {
        "experiment_name": "min_k_prob_attack_experiment",
        "base_model": "EleutherAI/pythia-160m",
        "dataset_member": "the_pile",
        "dataset_nonmember": "the_pile",
        "min_words": 100,
        "max_words": 200,
        "max_tokens": 512,
        "max_data": 100000,
        "output_name": "mink_prob_attack_results",
        "n_samples": 1000,
        "blackbox_attacks": ["min_k"],
        "env_config": {
            "results": "results_mink",
            "device": "cuda:0",
            "device_aux": "cuda:0"
        },
        "dump_cache": False,
        "load_from_cache": False,
        "load_from_hf": True,
        "batch_size": 50
    }
    run_minK_attack(mink_config, target_model, train, test, device)
    print("\n")

   # Loss Attack
    print("Running the Loss Attack")
    loss_config = {
        "experiment_name": "loss_attack_experiment",
        "base_model": "EleutherAI/pythia-160m",
        "dataset_member": "the_pile",
        "dataset_nonmember": "the_pile",
        "min_words": 100,
        "max_words": 200,
        "max_tokens": 512,
        "max_data": 100000,
        "output_name": "loss_attack_results",
        "n_samples": 1000,
        "blackbox_attacks": ["loss"],
        "env_config": {
            "results": "results_loss",
            "device": "cuda:0",
            "device_aux": "cuda:0"
        },
        "dump_cache": False,
        "load_from_cache": False,
        "load_from_hf": True,
        "batch_size": 50,
        "threshold": 2.6  # Adjust based on testing
    }
    run_loss_attack(loss_config, target_model, train, test)
    print("\n")



if __name__ == "__main__":
    # print(torch.__version__)
    print(torch.cuda.is_available())
    # print(torch.version.cuda)  # Should print a version like '12.2'
    # print(torch.backends.cudnn.enabled)  # Should print True if cuDNN is availa

    main()