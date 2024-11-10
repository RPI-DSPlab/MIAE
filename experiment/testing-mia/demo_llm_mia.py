import os
import torch
import json
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from experiment.llm_models import get_model
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import load_mimir_dataset
from miae.attacks.base import LLM_ModelAccess
from miae.attacks_on_llm.mink import MinKProbAttack, MinKAuxiliaryInfo


current_dir = os.getcwd()
attack_dir = os.path.join(current_dir, "attack")
savedir = os.path.join(current_dir, "results")
seed = 0

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
    document = test[0]['text']
    inputs = tokenizer(document, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    tokens = tokenizer.tokenize(document)

    # Now call _attack
    min_k_result = mink_attack._attack(document=document, probs=probs, tokens=tokens)
    print(f"the minK result is {min_k_result}")



if __name__ == "__main__":
    main()
