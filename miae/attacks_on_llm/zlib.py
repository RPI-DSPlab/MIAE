import numpy as np
import torch as ch
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
        # Compute model loss (likelihood)
        loss = self.target_model.get_signal_llm(
            text=document,
            tokens=tokens,
            no_grads=True,
            return_all_probs=False
        )['loss']

        # Compute zlib compression entropy of the document
        zlib_entropy = len(zlib.compress(bytes(document, "utf-8")))

        # Normalize the loss using zlib entropy
        zlib_normalized_score = loss / zlib_entropy

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
        self.device = config.get("device", 'cuda' if ch.cuda.is_available() else 'cpu')

    def save_config_to_dict(self):
        """
        Save the configuration to a dictionary.
        :return: Dictionary containing the configuration.
        """
        return self.config
