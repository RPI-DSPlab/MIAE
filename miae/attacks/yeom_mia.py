# This code implements the loss threshold based membership inference attack "Privacy Risk in Machine Learning:
# Analyzing the Connection to Overfitting".
# The code is based on the code from
# https://github.com/TinfoilHat0/MemberInference-by-LossThreshold
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from sklearn.metrics import roc_curve
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import dataset_split


class YeomAuxiliaryInfo(AuxiliaryInfo):
    """
    The auxiliary information for the Yeom attack.
    """

    def __init__(self, config, attack_model=None):
        """
        Initialize the auxiliary information with default config.
        :param config: a dictionary containing the configuration for auxiliary information.
        :param attack_model: the attack model architecture.
        """
        super().__init__(config)
        # ---- initialize auxiliary information with default values ----
        self.seed = config.get("seed", 0)
        self.batch_size = config.get("batch_size", 128)
        self.num_classes = config.get("num_classes", 10)
        self.num_classes = config.get("num_classes", 10)

        # -- other parameters --
        self.save_path = config.get("save_path", "yeom")
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)

        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.log_path is not None:
            self.logger = logging.getLogger('yeom_logger')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/yeom.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


class YeomModelAccess(ModelAccess):
    """
    Implementation of model access for Yeom attack.
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType = ModelAccessType.BLACK_BOX):
        """
        Initialize model access with model and access type.
        :param model: the target model.
        :param access_type: the type of access to the target model.
        """
        super().__init__(model, untrained_model, access_type)


class YeomUtil(MIAUtils):

    @classmethod
    def get_loss_n_accuracy(cls, model: YeomModelAccess, data_loader, num_classes, aux_info: YeomAuxiliaryInfo):
        """
        Returns loss/acc, and per-class loss/accuracy on supplied data loader
        model: model to evaluate
        data_loader: data loader to evaluate
        num_classes: number of classes in the dataset
        aux_info: auxiliary information of Yeom
        """

        with torch.inference_mode():
            total_loss, correctly_labeled_samples, num_sample = 0, 0, 0
            confusion_matrix = torch.zeros(num_classes, num_classes)
            per_class_loss = torch.zeros(num_classes, device=aux_info.device)
            per_class_ctr = torch.zeros(num_classes, device=aux_info.device)

            criterion = torch.nn.CrossEntropyLoss(reduction='none').to(aux_info.device)
            for _, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device=aux_info.device, non_blocking=True), \
                    labels.to(device=aux_info.device, non_blocking=True)
                model.to(aux_info.device)
                outputs = model(inputs)
                losses = criterion(outputs, labels)
                # keep track of total loss
                total_loss += losses.sum()
                num_sample += len(labels)
                # get num of correctly predicted inputs in the current batch
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()

                # per-class acc (filling confusion matrix)
                for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                # per-class loss
                for i in range(num_classes):
                    filt = labels == i
                    per_class_loss[i] += losses[filt].sum()
                    per_class_ctr[i] += filt.sum()

            loss = total_loss / num_sample
            loss = loss.cpu().item()
            accuracy = correctly_labeled_samples / len(data_loader.dataset)
            per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
            per_class_loss = per_class_loss / per_class_ctr

            return (loss, accuracy), (per_class_accuracy, per_class_loss)


class YeomAttack(MiAttack):
    """
    Implementation of the Yeom attack.
    """

    def __init__(self, target_model_access: YeomModelAccess, aux_info: YeomAuxiliaryInfo, target_data=None):
        """
        Initialize the Yeom attack with model access and auxiliary information.
        :param target_model_access: the model access to the target model.
        :param aux_info: the auxiliary information for the Shokri attack.
        :param target_data: the target data for the Shokri attack.
        """
        super().__init__(target_model_access, aux_info)
        self.aux_info = aux_info
        self.target_model_access = target_model_access
        self.threshold = None  # this is the loss threshold for the attack

        self.prepared = False  # this flag indicates whether the attack has been prepared

    def prepare(self, auxiliary_dataset):
        """
        Prepare the attack: train a shadow model to obtain the loss threshold
        :param auxiliary_dataset: the auxiliary dataset (will be split into training sets)
        """
        super().prepare(auxiliary_dataset)
        if self.prepared:
            print("The attack has already prepared!")
            return
        
        YeomUtil.log(self.aux_info, "Start preparing the attack...", print_flag=True)

        set_seed(self.aux_info.seed)

        # create directories:
        for path in [self.aux_info.save_path, self.aux_info.log_path]:
            if path is not None and not os.path.exists(path):
                os.makedirs(path)

        # get the loss threshold
        train_loader = DataLoader(auxiliary_dataset, batch_size=self.aux_info.batch_size, shuffle=False, num_workers=2)
        (self.threshold, _), _ = YeomUtil.get_loss_n_accuracy(self.target_model_access, train_loader,
                                                              self.aux_info.num_classes,
                                                              self.aux_info)

        YeomUtil.log(self.aux_info, f"Loss threshold: {self.threshold}", print_flag=True)

        YeomUtil.log(self.aux_info, "Finish preparing the attack...", print_flag=True)

        self.prepared = True

    def infer(self, target_data) -> np.ndarray:
        """
        Infer the membership of the target data.
        """
        super().infer(target_data)
        if not self.prepared:
            raise ValueError("The attack has not been prepared!")
        losses_threshold_diff = []

        YeomUtil.log(self.aux_info, "Start membership inference...", print_flag=True)

        # load the attack models
        target_data_loader = DataLoader(target_data, batch_size=self.aux_info.batch_size, shuffle=False, num_workers=2)
        self.target_model_access.to_device(self.aux_info.device)
        for data, _ in target_data_loader:
            data = data.to(self.aux_info.device)
            with torch.inference_mode():
                outputs = self.target_model_access.model(data)
                losses = F.cross_entropy(outputs, torch.argmax(outputs, dim=1), reduction='none')
                losses = losses.cpu().numpy()
                losses_threshold_diff += (losses - self.threshold).tolist()
        # mean and var of losses_threshold_diff
        mean = np.mean(losses_threshold_diff)
        var = np.var(losses_threshold_diff)
        YeomUtil.log(self.aux_info, f"Mean of losses_threshold_diff: {mean}", print_flag=True)

        # for the purpose of obtaining the prediction as a score, we couldn't just use the boolean value of the
        # losses_threshold_diff, but we need to normalize the value to [0, 1]
        min_diff, max_diff = min(losses_threshold_diff), max(losses_threshold_diff)
        losses_threshold_diff = np.array(losses_threshold_diff)
        losses_threshold_diff = (losses_threshold_diff - min_diff) / (max_diff - min_diff)

        predictions = 1 - losses_threshold_diff

        YeomUtil.log(self.aux_info, "Finish membership inference...", print_flag=True)

        return predictions
