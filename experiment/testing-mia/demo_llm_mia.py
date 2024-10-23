import torch
from datasets import load_dataset
from experiment import llm_models

# Load the dataset
cache_dir = "/data/public/huggingface_datasets"
dataset = load_dataset("iamgroot42/mimir", "pile_cc", split="ngram_7_0.2", cache_dir=cache_dir)

# Load the model
model_name = 'pythia_160m'
model, tokenizer = llm_models.get_model(model_name)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")


if __name__ == "__main__":
    main()
