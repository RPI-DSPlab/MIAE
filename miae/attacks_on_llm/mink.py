import numpy as np
import torch
from miae.attacks.base import ModelAccessType, AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack
from miae.attacks_on_llm.config import ExperimentConfig as ExpConfig

class MinKProbAttack(Attack):
    def __init__(self, config:ExpConfig, target_model, k: float = 0.2, window: int = 1, stride: int = 1, is_blackbox: bool = True):
        """
        Initialize MinKProbAttack with necessary parameters.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param k: Top-k percentage to consider.
        :param window: Window size for calculating n-gram probabilities.
        :param stride: Step size for the sliding window over n-grams.
        :param is_blackbox: Boolean indicating if this is a black-box attack.
        """
        super().__init__(config, target_model, is_blackbox)
        self.k = k
        self.window = window
        self.stride = stride

    def _attack(self, document, probs=None, tokens=None, **kwargs):
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

        ngram_probs = []

        # Calculate log prob for each n-gram based on the probs
        with torch.no_grad():
            for i in range(0, probs.shape[0] - self.window + 1, self.stride):
                ngram_prob = np.mean(probs[i:i + self.window].numpy())  # Average log prob for the window
                ngram_probs.append(ngram_prob)

        # Sort and select the top-k% probabilities (smallest values)
        min_k_probs = sorted(ngram_probs)[: int(len(ngram_probs) * self.k)]
        result = -np.mean(min_k_probs)

        return result

