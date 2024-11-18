import numpy as np
import torch
from miae.attacks.base import AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack

class MinKProbAttackPlusPlus(Attack):
    def __init__(self, target_model, config):
        """
        Initialize MinKProbAttackPlusPlus with additional enhancements.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param k: Top-k percentage to consider.
        :param window: Window size for calculating n-gram probabilities.
        :param stride: Step size for the sliding window over n-grams.
        :param is_blackbox: Boolean indicating if this is a black-box attack.
        """
        super().__init__(config, target_model)
        self.k = config.get("k", 0.2)
        self.window = config.get("window", 5)
        self.stride = config.get("stride", 1)
        self.is_blackbox = config.get("is_blackbox", True)
        self.threshold = None

    @torch.no_grad()
    def _attack(self, document, probs=None, tokens=None):
        """
        Enhanced Min-K%++ Prob Attack using normalized n-gram probabilities.

        :param document: The input text document for which to calculate probabilities.
        :param probs: Precomputed probabilities, if available.
        :param tokens: Optional token IDs if available.

        :return: Negative average log probability of the normalized top-k% n-grams.
        """
        all_prob = (
            probs
            if probs is not None
            else self.target_model.get_signal_llm(text=document, tokens=tokens, no_grads=True, return_all_probs=True)
        )
        if isinstance(all_prob, torch.Tensor):
            all_prob = all_prob.cpu().numpy()

        # Compute n-gram probabilities with normalization
        ngram_probs = []
        for i in range(0, len(all_prob) - self.window + 1, self.stride):
            ngram_prob = all_prob[i:i + self.window]
            mean_prob = np.mean(ngram_prob)
            std_prob = np.std(ngram_prob)
            normalized_prob = (mean_prob - np.mean(all_prob)) / (std_prob + 1e-8)
            ngram_probs.append(normalized_prob)

        # Consider variance information
        variances = [np.var(all_prob[i:i + self.window]) for i in range(0, len(all_prob) - self.window + 1, self.stride)]
        combined_scores = [prob + var for prob, var in zip(ngram_probs, variances)]

        k_count = max(1, int(len(combined_scores) * self.k))
        min_k_scores = sorted(combined_scores)[:k_count]

        return -np.mean(min_k_scores)

    def find_optimal_threshold(self, scores, labels):
        """
        Advanced threshold determination using cross-validation.

        :param scores: List of Min-K%++ scores.
        :param labels: List of labels (1 for member, 0 for non-member).
        :return: The optimal threshold value.
        """
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5)
        best_threshold = 0.0
        best_accuracy = 0.0

        for train_index, val_index in skf.split(scores, labels):
            for threshold in np.linspace(min(scores), max(scores), 100):
                predictions = [1 if score > threshold else 0 for score in np.array(scores)[val_index]]
                accuracy = np.mean([pred == label for pred, label in zip(predictions, np.array(labels)[val_index])])
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold

        return best_threshold

    def get_threshold(self, train_set, device):
        """
        Determine the optimal threshold using a validation-based method.

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

            # Calculate the Min-K%++ probability score
            score = self._attack(document=document['text'], probs=probs)
            scores.append(score)
            labels.append(1)

        best_threshold = self.find_optimal_threshold(scores, labels)
        self.threshold = best_threshold
        return best_threshold


class MinKAuxiliaryInfoPlusPlus(AuxiliaryInfo):
    def __init__(self, config):
        super().__init__(config)

    def save_config_to_dict(self):
        return super().save_config_to_dict()
