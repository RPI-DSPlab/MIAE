from transformers import GPTNeoXForCausalLM, AutoTokenizer


# Get models
def get_model(model_name, device):
    if model_name == 'pythia_160m':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-160m",
            revision="step3000",
            cache_dir="./pythia-160m/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m",
            revision="step3000",
            cache_dir="./pythia-160m/step3000",
        )
    elif model_name == 'pythia_1_4b':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-1.4b",
            revision="step3000",
            cache_dir="./pythia-1_4b/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-1.4b",
            revision="step3000",
            cache_dir="./pythia-1_4b/step3000",
        )
    elif model_name == 'pythia_2_8b':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-2.8b",
            revision="step3000",
            cache_dir="./pythia-2_8b/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-2.8b",
            revision="step3000",
            cache_dir="./pythia-2_8b/step3000",
        )
    elif model_name == 'pythia_6_9b':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-6.9b",
            revision="step3000",
            cache_dir="./pythia-6_9b/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-6.9b",
            revision="step3000",
            cache_dir="./pythia-6_9b/step3000",
        )
    elif model_name == 'pythia_12b':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-12b",
            revision="step3000",
            cache_dir="./pythia-12b/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-12b",
            revision="step3000",
            cache_dir="./pythia-12b/step3000",
        )
    elif model_name == 'pythia_160m_deduped':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-160m-deduped",
            revision="step3000",
            cache_dir="./pythia-160m-deduped/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m-deduped",
            revision="step3000",
            cache_dir="./pythia-160m-deduped/step3000",
        )
    elif model_name == 'pythia_1_4b_deduped':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-1.4b-deduped",
            revision="step3000",
            cache_dir="./pythia-1_4b_deduped/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-1.4b-deduped",
            revision="step3000",
            cache_dir="./pythia-1_4b_deduped/step3000",
        )
    elif model_name == 'pythia_2_8b_deduped':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-2.8b-deduped",
            revision="step3000",
            cache_dir="./pythia-2_8b_deduped/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-2.8b-deduped",
            revision="step3000",
            cache_dir="./pythia-2_8b_deduped/step3000",
        )
    elif model_name == 'pythia_6_9b_deduped':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-6.9b-deduped",
            revision="step3000",
            cache_dir="./pythia-6_9b_deduped/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-6.9b-deduped",
            revision="step3000",
            cache_dir="./pythia-6_9b_deduped/step3000",
        )
    elif model_name == 'pythia_12b_deduped':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-12b-deduped",
            revision="step3000",
            cache_dir="./pythia-12b_deduped/step3000",
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-12b-deduped",
            revision="step3000",
            cache_dir="./pythia-12b_deduped/step3000",
        )
    else:
        raise ValueError('Unknown model name')
    return model, tokenizer

