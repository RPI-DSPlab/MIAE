import numpy as np
import torch
from miae.attacks.base import ModelAccessType, AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack


class LossAttack(Attack):
    def __init__(self, target_model, config):
        """
        Initialize LossAttack with necessary parameters.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param config: Dictionary containing configuration parameters.
        """
        super().__init__(config, target_model)
        self.is_blackbox = config.get("is_blackbox", True)

    def _attack(self, document, tokens=None):
        """
        Main logic for the Loss Attack. Computes model loss and returns it as the MIA score.

        :param document: The input text document for which to calculate the loss.
        :param tokens: Optional token IDs if available.

        :return: Continuous MIA score (likelihood score) based on model loss.
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
        # print(f"Average Loss (Likelihood Score): {avg_loss}")
        # print(f"Normalized Probability-Like Score: {np.exp(-avg_loss)}")

        # Return the continuous likelihood score directly
        return np.exp(-avg_loss)
        # return avg_loss  # Higher score indicates lower likelihood, i.e., more likely non-member


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
        self.config = config
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

    def save_config_to_dict(self):
        """
        Save the configuration to a dictionary.
        :return: Dictionary containing the configuration.
        """
        return self.config
