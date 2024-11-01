import numpy as np
import torch
from miae.attacks.base import ModelAccessType, AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack

class LossAttack(Attack):
    def __init__(self, target_model, config):
        """
        Initialize LossAttack with necessary parameters.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param config: Dictionary containing configuration parameters, including threshold.
        """
        super().__init__(config, target_model)
        self.threshold = config.get("threshold", None)
        self.is_blackbox = config.get("is_blackbox", True)

        if self.threshold is None:
            raise ValueError("Threshold not set in configuration for LossAttack")

    def _attack(self, document, tokens=None):
        """
        Main logic for the Loss Attack. Computes model loss and returns a membership prediction based on the threshold.

        :param document: The input text document for which to calculate the loss.
        :param tokens: Optional token IDs if available.

        :return: Binary prediction (1 if member, 0 if non-member).
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
        
        # Return membership prediction based on the threshold
        return int(avg_loss < self.threshold)  # 1 if member, 0 if non-member

class LossAttackAuxiliaryInfo(AuxiliaryInfo):
    """
    Class to encapsulate the configuration for the Loss Attack.
    """
    def __init__(self, config):
        """
        Initialize the LossAttackAuxiliaryInfo with the provided configuration dictionary.
        :param config: Dictionary containing configuration parameters.
        """
        super().__init__(config)
        self.threshold = config.get("threshold", None)
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

    def save_config_to_dict(self):
        """
        Save the configuration to a dictionary.
        :return: Dictionary containing the configuration.
        """
        return super().save_config_to_dict()
