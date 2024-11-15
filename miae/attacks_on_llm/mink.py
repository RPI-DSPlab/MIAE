import numpy as np
import torch
from miae.attacks.base import AuxiliaryInfo
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
        self.window = config.get("window", 5)  # Default window size
        self.stride = config.get("stride", 1)  # Default stride
        self.is_blackbox = config.get("is_blackbox", True)
        self.threshold = None

    @torch.no_grad()
    def _attack(self, document, probs=None, tokens=None):
        """
        Min-k % Prob Attack using window and stride to analyze n-grams.

        :param document: The input text document for which to calculate probabilities.
        :param probs: Precomputed probabilities, if available.
        :param tokens: Optional token IDs if available.

        :return: Negative average log probability of the top-k% n-grams.
        """
        all_prob = (
            probs
            if probs is not None
            else self.target_model.get_signal_llm(text=document, tokens=tokens, no_grads=True, return_all_probs=True)
        )
        # Convert all_prob to a NumPy array if necessary
        if isinstance(all_prob, torch.Tensor):
            all_prob = all_prob.cpu().numpy()

        ngram_probs = []
        for i in range(0, len(all_prob) - self.window + 1, self.stride):
            ngram_prob = all_prob[i:i + self.window]
            ngram_probs.append(np.mean(ngram_prob))

        k_count = max(1, int(len(ngram_probs) * self.k))
        min_k_probs = sorted(ngram_probs)[:k_count]

        return -np.mean(min_k_probs)

    def find_optimal_threshold(self, scores, labels):
        """
        Find the optimal threshold that maximizes detection accuracy.

        :param scores: List of Min-K% scores.
        :param labels: List of labels (1 for member, 0 for non-member).
        :return: The optimal threshold value.
        """
        # Example approach: Choose a threshold that maximizes accuracy
        best_threshold = 0.0
        best_accuracy = 0.0

        for threshold in np.linspace(min(scores), max(scores), 100):
            predictions = [1 if score > threshold else 0 for score in scores]
            accuracy = np.mean([pred == label for pred, label in zip(predictions, labels)])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return best_threshold

    def get_threshold(self, train_set, device):
        """
        Determine the optimal threshold by maximizing detection accuracy on the training set.

        :param train_set: The dataset to use for threshold calibration.
        :param device: The device to perform computations on.
        :return: The optimal threshold value.
        """
        scores = []
        labels = []

        for document in train_set:
            log_probs_data = self.target_model.get_signal_llm(
                text=document['text'],
                no_grads=True,
                return_all_probs=True
            )
            probs = torch.tensor(log_probs_data['all_token_log_probs'], device=device)

            # Calculate the Min-K% probability score
            score = self._attack(document=document['text'], probs=probs)
            scores.append(score)
            labels.append(1)  

        best_threshold = self.find_optimal_threshold(scores, labels)
        self.threshold = best_threshold
        return best_threshold


class MinKAuxiliaryInfo(AuxiliaryInfo):
    def __init__(self, config):
        super().__init__(config)

    def save_config_to_dict(self):
        return super().save_config_to_dict()
