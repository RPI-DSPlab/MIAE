# This code implements the loss trajectory based membership inference attack, published in CCS 2022 "Membership
# Inference Attacks by Exploiting Loss Trajectory".
# The code is based on the code from
# https://github.com/DennisLiu2022/Membership-Inference-Attacks-by-Exploiting-Loss-Trajectory
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack

class LosstrajAuxiliaryInfo(AuxiliaryInfo):
    """
    The auxiliary information for the loss trajectory based membership inference attack.
    """
    def __init__(self, config: dict):
        """
        Initialize the auxiliary information.
        :param config: the loss trajectory.
        """
        super().__init__(config)
        self.loss_trajectory = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.get('seed', 0)
        self.batch_size = config.get('batch_size', 128)
        self.num_workers = config.get('num_workers', 2)
        self.num_epochs = config.get('num_epochs', 240)

        # directories:
        self.save_dir = config.get('save_dir', './results')
        self.distill_models_






class LosstrajModelAccess(ModelAccess):
    """
    The model access for the loss trajectory based membership inference attack.
    """
    def __init__(self, model, model_type):
        """
        Initialize the model access.
        :param model: the target model.
        :param model_type: the type of the target model.
        """
        super().__init__(model, model_type)

    def get_model_architecture(self):
        """
        Get the un-initialized model architecture of the target model.
        :return: the un-initialized model architecture.
        """
        return copy.deepcopy(self.model).reset_parameters()

class LosstrajUtil:
    @classmethod
    def model_distillation(cls, target_model_access: LosstrajModelAccess, distillation_dataset: DataLoader, auxiliary_info: LosstrajAuxiliaryInfo):
        """
         with distillation_dataset.
        :param target_model_access:
        :param dataset:
        :param auxiliary_info:
        :return:
        """

        distilled_model = target_model_access.get_model_architecture()
        distilled_model.to(auxiliary_info.device)
        distilled_model.train()

        # obtain the distillation dataset from target model
        distillation_loader = distillation_dataset

class LosstrajAttack(MiAttack):
    """
    Implementation of Losstraj attack.
    """

    def __init__(self, target_model_access: LosstrajModelAccess, auxiliary_info: LosstrajAuxiliaryInfo):
        """
        Initialize the attack.
        :param target_model_access: the target model access.
        :param auxiliary_info: the auxiliary information.
        """
        super().__init__(target_model_access, auxiliary_info)

    def prepare(self, shadow_set):
        """
        Prepare the attack.
        """
