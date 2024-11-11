import os
import torch
import json
import numpy as np
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


def get_threshold(train_set, test_set, attack, tokenizer, device):
    member_scores = []
    non_member_scores = []

    # Compute scores for the training set (members)
    print(f"the length of the train set is: {len(train_set)}")
    c1 = 0
    for document in train_set:
        if c1 % 100 == 0:
            print(f"c1 is {c1}")
        log_probs_data = attack.target_model.get_signal_llm(
            text=document['text'],
            no_grads=True,
            return_all_probs=True
        )
        probs = torch.tensor(log_probs_data['all_token_log_probs'], device=device)
        tokens = tokenizer.tokenize(document['text'])
        score = attack._attack(document=document['text'], probs=probs, tokens=tokens)
        member_scores.append(score)
        c1 += 1

    # Compute scores for the test set (non-members)
    c2 = 0
    print(f"the length of the test set is: {len(test_set)}")
    for document in test_set:
        if c2 % 100 == 0:
            print(f"c2 is {c2}")
        log_probs_data = attack.target_model.get_signal_llm(
            text=document['text'],
            no_grads=True,
            return_all_probs=True
        )
        probs = torch.tensor(log_probs_data['all_token_log_probs'], device=device)
        tokens = tokenizer.tokenize(document['text'])
        score = attack._attack(document=document['text'], probs=probs, tokens=tokens)
        non_member_scores.append(score)
        c2 += 1

    avg_member_score = np.mean(member_scores)
    avg_non_member_score = np.mean(non_member_scores)
    threshold = (avg_member_score + avg_non_member_score) / 2

    return threshold


def main():
    # set up
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # Load the model
    model_name = 'pythia_160m'
    model, tokenizer = get_model(model_name)
    # Initialize the model access
    target_model = LLM_ModelAccess(model, tokenizer, device)

    # Load the dataset
    cache_dir = "/data/public/huggingface_datasets"
    split_dateset = load_mimir_dataset("arxiv", "ngram_13_0.8", cache_dir, test_size=0.5, seed=1)
    train = split_dateset['train']
    test = split_dateset['test']

    # Attack the model
    
    print("Running the minK attack")
    
    # Default configuration for minK attack
    default_config = {
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
    mink_info = MinKAuxiliaryInfo(default_config)
    min_info_dict = mink_info.save_config_to_dict()
    mink_attack = MinKProbAttack(target_model, min_info_dict)

    threshold = get_threshold(train, test, mink_attack, tokenizer, device)
    print(f"the threshold is {threshold}")

    document = test[0]['text']
    inputs = tokenizer(document, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    tokens = tokenizer.tokenize(document)

    # Now call _attack
    min_k_result = mink_attack._attack(document=document, probs=probs, tokens=tokens)
    print(f"The minK result is: {min_k_result} \n")

    if min_k_result < threshold:
        print("The document is classified as a member.")
    else:
        print("The document is classified as a non-member.")

    print("Running the loss attack")

    # Attack configuration for LossAttack
    loss_attack_config = {
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
    
    loss_info = LossAttackAuxiliaryInfo(loss_attack_config)
    config_dict = loss_info.save_config_to_dict()
    
    # Initialize LossAttack
    loss_attack = LossAttack(target_model, config_dict)

    # Select a document from the test dataset
    document = test[0]['text']
    inputs = tokenizer(document, return_tensors="pt").to(device)

    # Get token IDs instead of strings
    tokens = tokenizer.encode(document)

    # Run the LossAttack
    result = loss_attack._attack(document=document, tokens=tokens)
    print(f"Threshold: {loss_attack.threshold}")
    print(f"The loss membership inference result is: {'member' if result == 1 else 'non-member'}")

if __name__ == "__main__":
    main()