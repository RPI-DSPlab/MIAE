import os
import torch
import json
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from experiment.llm_models import get_model
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import load_mimir_dataset
from miae.attacks.base import LLM_ModelAccess
from miae.attacks_on_llm.loss import LossAttack, LossAttackAuxiliaryInfo


current_dir = os.getcwd()
attack_dir = os.path.join(current_dir, "attack")
savedir = os.path.join(current_dir, "results")
seed = 0

def main():
    # Set up
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model_name = 'pythia_160m'
    model, tokenizer = get_model(model_name)
    target_model = LLM_ModelAccess(model, tokenizer, device)    
    
    # Load the dataset
    cache_dir = "/data/public/huggingface_datasets"
    split_dataset = load_mimir_dataset("arxiv", "ngram_13_0.8", cache_dir, test_size=0.5, seed=1)
    train = split_dataset['train']
    test = split_dataset['test']

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
    print(f"The membership inference result is: {'member' if result == 1 else 'non-member'}")


if __name__ == "__main__":
    main()
