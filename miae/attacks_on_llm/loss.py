import logging
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from miae.attacks.base import AuxiliaryInfo, LLM_ModelAccess, MIAUtils, MiAttack


class LossAttack_AuxiliaryInfo(AuxiliaryInfo):
    """
    The auxiliary information for the LLM loss attack.
    """
    def __init__(self, config):
        super().__init__(config)
        self.seed = config.get("seed", 0)
        self.batch_size = config.get("batch_size", 32)
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = config.get("num_classes", None)  # Specific for classification tasks
        self.log_path = config.get("log_path", None)
        
        if self.log_path:
            self.logger = logging.getLogger('llm_loss_attack_logger')
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_path + '/llm_loss_attack.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(handler)


class LossAttack_ModelAccess(LLM_ModelAccess):
    """
    Access the LLM model for the loss attack or log probability-based attack.
    """
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def get_log_probs(self, text):
        """
        Use `get_signal_llm` to retrieve the log probabilities for a given text.
        """
        log_probs = self.get_signal_llm(text=text, no_grads=True, return_all_probs=True)
        return log_probs

    def get_loss(self, text):
        """
        Compute the loss using the LLM for the given input text (similar to `get_signal_llm`, but using built-in loss).
        """
        self.model.eval()  # Set the model to evaluation mode
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens, labels=tokens["input_ids"])
            loss = outputs.loss.item()
        
        return loss


class LossAttack(MiAttack):
    """
    Implementation of the loss-based membership inference attack using precomputed log probabilities from a pickle file.
    """

    def __init__(self, target_model_access: LLM_ModelAccess, aux_info: LossAttack_AuxiliaryInfo, target_data=None):
        super().__init__(target_model_access, aux_info)
        self.aux_info = aux_info
        self.target_model_access = target_model_access
        self.threshold = None
        self.prepared = False

    def prepare(self, auxiliary_dataset):
        # You may set the threshold here after processing auxiliary_dataset
        pass

    def load_log_probs(self, pickle_filename):
        """
        Load log probabilities from a pickle file.
        :param pickle_filename: The path to the pickle file containing the log probabilities.
        :return: The loaded log probabilities.
        """
        with open(pickle_filename, 'rb') as f:
            log_prob_data = pickle.load(f)
        return log_prob_data

    def infer(self, data, pickle_filename='log_probs.pkl'):
        """
        Infer the membership of the target data using precomputed log probabilities from the pickle file.
        :param data: The target data to infer membership for.
        :param pickle_filename: The filename of the pickle file containing log probabilities.
        """
        log_prob_data = self.load_log_probs(pickle_filename)
        target_token_log_probs = log_prob_data['target_token_log_probs']

        # Inference logic based on log probabilities
        predictions = []
        for log_prob in target_token_log_probs:
            # Use a comparison to a threshold, or another method, to infer membership
            # Here, you may decide whether to sum or average the log probabilities before comparing
            if log_prob > self.threshold:
                predictions.append(1)  # In
            else:
                predictions.append(0)  # Out

        return np.array(predictions)


class LossAttack_Utils(MIAUtils):
    @classmethod
    def compute_average_loss(cls, model_access, data_loader, aux_info):
        """
        Compute the average loss on a dataset.
        """
        total_loss = 0
        total_samples = 0
        
        for texts in data_loader:
            loss = model_access.get_loss(texts)
            total_loss += loss
            total_samples += len(texts)  # Ensure len(texts) works as expected
        
        return total_loss / total_samples
    
    @classmethod
    def compute_average_log_probs(cls, model_access, data_loader, aux_info):
        """
        Compute the average log probabilities on a dataset.
        """
        total_log_probs = 0
        total_samples = 0
        
        for texts in data_loader:
            log_probs = model_access.get_log_probs(texts)
            total_log_probs += sum(log_probs['target_token_log_probs'])  # Ensure log_probs is aggregated
            total_samples += len(texts)
        
        return total_log_probs / total_samples
