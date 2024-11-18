import numpy as np
import torch
import zlib
from miae.attacks.base import ModelAccessType, AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack

class ZLIBAttack(Attack):
    """
    ZLIB Attack: Normalizes model likelihood (loss) with zlib compression entropy of the document.
    """

    def __init__(self, target_model, config):
        """
        Initialize ZLIBAttack with necessary parameters.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param config: Dictionary containing configuration parameters.
        """
        super().__init__(config, target_model)
        self.is_blackbox = config.get("is_blackbox", True)

    def _attack(self, document, tokens=None):
        """
        Main logic for the ZLIB Attack. Normalizes the likelihood score using zlib compression entropy.

        :param document: The input text document.
        :param tokens: Optional token IDs if available.

        :return: ZLIB-normalized likelihood score.
        """
         # Retrieve log probabilities for each token in the document
        log_probs_data = self.target_model.get_signal_llm(
            text=document,
            tokens=tokens,
            no_grads=True,
            return_all_probs=False
        )
        target_token_log_probs = log_probs_data['target_token_log_probs']

        # Convert log probabilities to losses (negative log likelihoods)
        with torch.no_grad():
            losses = [-log_prob for log_prob in target_token_log_probs]

        # Calculate the average loss over the document
        avg_loss = np.mean(losses)
        
        # Compute zlib compression entropy of the document
        zlib_entropy = len(zlib.compress(bytes(document, "utf-8")))

        # Normalize the loss using zlib entropy
        zlib_normalized_score = avg_loss / zlib_entropy

        return zlib_normalized_score


class ZLIBAttackAuxiliaryInfo(AuxiliaryInfo):
    """
    Class to encapsulate the configuration for the ZLIB Attack.
    """

    def __init__(self, config):
        """
        Initialize the ZLIBAttackAuxiliaryInfo with the provided configuration dictionary.

        :param config: Dictionary containing configuration parameters.
        """
        super().__init__(config)
        self.config = config
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

    def save_config_to_dict(self):
        """
        Save the configuration to a dictionary.
        :return: Dictionary containing the configuration.
        """
        return self.config
