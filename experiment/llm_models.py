import os
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Get models
def get_model(model_name, device):
    # Define base cache directory
    base_cache_dir = "/data/public/llm_mia_comp/models"

    # Define the specific cache directory based on the model name
    model_cache_dirs = {
        'pythia_160m': f"{base_cache_dir}/pythia-160m/step3000",
        'pythia_1_4b': f"{base_cache_dir}/pythia-1_4b/step3000",
        'pythia_2_8b': f"{base_cache_dir}/pythia-2_8b/step3000",
        'pythia_6_9b': f"{base_cache_dir}/pythia-6_9b/step3000",
        'pythia_12b': f"{base_cache_dir}/pythia-12b/step3000",
        'pythia_160m_deduped': f"{base_cache_dir}/pythia-160m-deduped/step3000",
        'pythia_1_4b_deduped': f"{base_cache_dir}/pythia-1_4b_deduped/step3000",
        'pythia_2_8b_deduped': f"{base_cache_dir}/pythia-2_8b_deduped/step3000",
        'pythia_6_9b_deduped': f"{base_cache_dir}/pythia-6_9b_deduped/step3000",
        'pythia_12b_deduped': f"{base_cache_dir}/pythia-12b_deduped/step3000"
    }

    # Get the cache directory for the specified model name
    cache_dir = model_cache_dirs.get(model_name)
    if not cache_dir:
        raise ValueError('Unknown model name')

    os.makedirs(cache_dir, exist_ok=True)

    # Load the model
    if model_name in model_cache_dirs:
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{model_name.replace('_', '-')}",
            revision="step3000",
            cache_dir=cache_dir
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            f"EleutherAI/{model_name.replace('_', '-')}",
            revision="step3000",
            cache_dir=cache_dir
        )
    else:
        raise ValueError('Unknown model name')

    return model, tokenizer