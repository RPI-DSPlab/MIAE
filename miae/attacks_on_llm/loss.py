import numpy as np
import torch
from miae.attacks.base import ModelAccessType, AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack

class LossAttack(Attack):
    def __init__(self, config, target_model, threshold: float = None, is_blackbox: bool = True):
        """
        Initialize LossAttack with necessary parameters.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param threshold: Loss threshold to infer membership.
        :param is_blackbox: Boolean indicating if this is a black-box attack.
        """
        super().__init__(config, target_model, is_blackbox)
        self.threshold = threshold

    def _attack(self, document, tokens=None, **kwargs):
        """
        Main logic for the Loss Attack. Computes model loss and returns a membership prediction based on the threshold.

        :param document: The input text document for which to calculate the loss.
        :param tokens: Optional token IDs if available.

        :return: Binary prediction (1 if member, 0 if non-member).
        """
        # Calculate log probabilities and loss for the document
        log_probs_data = self.target_model.get_signal_llm(
            text=document,
            tokens=tokens,
            no_grads=True,
            return_all_probs=False
        )
        target_token_log_probs = log_probs_data['target_token_log_probs']

        # Convert log probabilities to loss (negative log likelihood)
        with torch.no_grad():
            losses = [-log_prob for log_prob in target_token_log_probs]
        
        # Compute average loss over the document
        avg_loss = np.mean(losses)
        
        # Infer membership based on the threshold
        if self.threshold is not None:
            return int(avg_loss < self.threshold)  # 1 if member, 0 if non-member
        else:
            raise ValueError("Threshold not set for LossAttack")

class LossAttackAuxiliaryInfo(AuxiliaryInfo):
    def __init__(self, config):
        """
        Initialize the auxiliary information.
        :param config: the configuration dictionary.
        """
        super().__init__(config)
        self.threshold = config.get("threshold", None)
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
