import numpy as np
import torch
import zlib
from miae.attacks.base import AuxiliaryInfo
from miae.attacks_on_llm.all_attacks import Attack


class ZLIBAttack(Attack):
    def __init__(self, target_model, config):
        """
        Initialize ZLIBAttack with necessary parameters.

        :param target_model: Instance of LLM_ModelAccess (Model) for the target model.
        :param config: Dictionary containing configuration parameters, including threshold and compression level.
        """
        super().__init__(config, target_model)
        self.threshold = config.get("threshold", None)
        self.compression_level = config.get("compression_level", 6)  # Default level for ZLIB compression
        self.is_blackbox = config.get("is_blackbox", True)

        if self.threshold is None:
            raise ValueError("Threshold not set in configuration for ZLIBAttack")

    def _attack(self, document, tokens=None):
        """
        Main logic for the ZLIB Attack. Computes model loss, combines it with ZLIB compression ratio, and returns
        a membership prediction based on the threshold.

        :param document: The input text document for which to calculate the loss and compression ratio.
        :param tokens: Optional token IDs if available.

        :return: Binary prediction (1 if member, 0 if non-member).
        """
        # Step 1: Calculate log probabilities (loss) for each token in the document
        log_probs_data = self.target_model.get_signal_llm(
            text=document,
            tokens=tokens,
            no_grads=True,
            return_all_probs=False
        )
        target_token_log_probs = log_probs_data['target_token_log_probs']

        with torch.no_grad():
            losses = [-log_prob for log_prob in target_token_log_probs]

        avg_loss = np.mean(losses)

        # Step 2: Apply ZLIB compression to the document text
        compressed_document = zlib.compress(document.encode('utf-8'), level=self.compression_level)
        compression_ratio = len(compressed_document) / len(document.encode('utf-8'))

        # Step 3: Combine average loss and compression ratio
        combined_metric = avg_loss * compression_ratio

        # Step 4: Return membership prediction based on the threshold
        return int(combined_metric < self.threshold)  # 1 if member, 0 if non-member


class ZLIBAuxiliaryInfo(AuxiliaryInfo):
    """
    Class to encapsulate the configuration for the ZLIB Attack.
    """

    def __init__(self, config):
        """
        Initialize the ZLIBAuxiliaryInfo with the provided configuration dictionary.
        :param config: Dictionary containing configuration parameters.
        """
        super().__init__(config)
        self.threshold = config.get("threshold", None)
        self.compression_level = config.get("compression_level", 6)
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

    def save_config_to_dict(self):
        """
        Save the configuration to a dictionary.
        :return: Dictionary containing the configuration.
        """
        return super().save_config_to_dict()
