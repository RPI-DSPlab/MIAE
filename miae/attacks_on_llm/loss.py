import logging
import os

import numpy as np
import torch

from miae.attacks.base import AuxiliaryInfo, LLM_ModelAccess


class LossAttack_AuxiliaryInfo(AuxiliaryInfo):
     """
    The auxiliary information for the loss attack.
    """
    
    def __init__(self, config, attack_model=None):
        # Todo
        super().__init__(config)
        
        
class LossAttack_Access(LLM_ModelAccess):
    """
    Implementation of model access for loss attack.
    """
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize model access with model and access type.
        :param model: the target model.
        :param access_type: the type of access to the target model.
        """
        super().__init__(self, model, tokenizer, device)


class LossAttack(MIAttack):
    """
    Implementation of the loss attack.
    """
    
    def __init__(self, target_model_access: LLM_ModelAccess, aux_info: LossAttack_AuxiliaryInfo, target_data=None)):
        # Todo
        
    def prepare(self, auxiliary_dataset):
        # Todo
        
        
class LossAttack_Utils(MIAUtils):
    """
    Implementation of loss attack utils.
    """
    