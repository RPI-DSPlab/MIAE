import numpy as np
import torch
from miae.attacks.base import ModelAccessType, AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack

class MinKProbAttack(Attack):
    def __init__(self, target_model, config):
        """
        Initialize MinKProbAttack with necessary parameters.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param k: Top-k percentage to consider.
        :param window: Window size for calculating n-gram probabilities.
        :param stride: Step size for the sliding window over n-grams.
        :param is_blackbox: Boolean indicating if this is a black-box attack.
        """
        super().__init__(config, target_model)
        self.k = config.get("k", 0.2)
        self.window = config.get("window", 1)
        self.stride = config.get("stride", 1)
        self.is_blackbox = config.get("is_blackbox", True)

    def _attack(self, document, probs=None, tokens=None):
        """
        Main logic for the Min-k % Prob Attack. Computes model probabilities and returns likelihood over the top k% of n-grams.

        :param document: The input text document for which to calculate probabilities.
        :param probs: Precomputed probabilities, if available.
        :param tokens: Optional token IDs if available.

        :return: Negative average log probability of the top-k% n-grams.
        """
        # Use the provided probs if available; otherwise, calculate using get_signal_llm
        if probs is None:
            log_probs_data = self.target_model.get_signal_llm(
                text=document,
                tokens=tokens,
                no_grads=True,
                return_all_probs=True
            )
            probs = log_probs_data['all_token_log_probs']

        probs = probs.squeeze(0).max(dim=-1).values
        ngram_probs = []

        # Calculate log prob for each n-gram based on the probs
        with torch.no_grad():
            for i in range(0, len(probs) - self.window + 1, self.stride):
                ngram_prob = np.mean(probs[i:i + self.window].numpy())  # Average log prob for the window
                ngram_probs.append(ngram_prob)

        min_k_probs = sorted(ngram_probs)[: int(len(ngram_probs) * self.k)]
        result = -np.mean(min_k_probs)

        return result

class MinKAuxiliaryInfo(AuxiliaryInfo):
    """
    Class to encapsulate the configuration for the Min-K Probability Attack.
    """
    def __init__(self, config):
        """
        Initialize the MinKAuxiliaryInfo with the provided configuration dictionary.
        :param config: Dictionary containing the configuration parameters.
        """
        super().__init__(config)


    def save_config_to_dict(self):
        """
        Save the configuration to a dictionary.
        :return: Dictionary containing the configuration.
        """
        return super().save_config_to_dict()