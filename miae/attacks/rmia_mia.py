# This code implements RMIA from "Low-Cost High-Power Membership Inference Attacks" by Sajjad et al.
# The code is based on the code from
# https://arxiv.org/abs/2312.03262
# Since Sajjad et al. , similar to Ye et al. implemented their Attack-R based on the code for LIRA attack,
# we will be reusing many components from the LIRA attack code.

import copy
import logging
import os
import re
from typing import List, Tuple

import numpy as np
import scipy
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from tqdm import tqdm
from scipy.stats import trim_mean

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from miae.utils.dataset_utils import get_xy_from_dataset
from miae.utils.set_seed import set_seed

from miae.attacks.lira_mia import LIRAUtil


class RMIAModelAccess(ModelAccess):
    """
    Your implementation of ModelAccess for RMIA.
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType = ModelAccessType.BLACK_BOX):
        """
        Initialize LiraModelAccess.
        """
        super().__init__(model, untrained_model, access_type)
        self.model = model
        self.model.eval()

    def get_signal_rmia(self, dataloader, device):
        """
        Get the signal of the model on the given dataset, a wrapper function
        to call the get_signal_lira function from from base class.

        :param dataloader: The dataloader for the dataset.
        :param device: The device to run the model on.
        :return: The signal of the model on the dataset.
        """
        return self.get_signal_lira(dataloader, device, augmentation=18)


class RMIAAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of AuxiliaryInfo for RMIA.
    """

    def __init__(self, config):
        """
        Initialize RMIAAuxiliaryInfo with a configuration dictionary.
        """
        super().__init__(config)
        self.config = config

        # Training parameters
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.lr = config.get('lr', 0.1)
        self.momentum = config.get('momentum', 0.9)
        self.decay = config.get('decay', 0.9999)
        self.shadow_seed_base = config.get('shadow_seed_base', 100)  # the seed begin number for shadow model
        self.epochs = config.get('epochs', 100)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('shadow_batchsize', 128)

        # Model saving and loading parameters
        self.save_path = config.get('save_path', None)

        # Auxiliary info for RMIA
        self.num_shadow_models = config.get('num_shadow_models', 8)
        self.shadow_path = config.get('shadow_path', f"{self.save_path}/weights/shadow/")
        self.online = config.get('online', True)
        self.nb_augmentations = config.get('nb_augmentations', 18)
        self.gamma = config.get('gamma', 0.5)

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)

        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.log_path is not None:
            self.lira_logger = logging.getLogger('rmia_logger')
            self.lira_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/rmia.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.lira_logger.addHandler(fh)


def _split_data(fullset, expid, iteration_range):
    keep = np.random.uniform(0, 1, size=(iteration_range, len(fullset)))
    order = keep.argsort(0)
    keep = order < int(.5 * iteration_range)
    keep = np.array(keep[expid], dtype=bool)
    return np.where(keep)[0], np.where(~keep)[0]


class RMIAUtil:
    """
    RMIA shares many of the code with LIRA attack, so we only define methods unique to RMIA here.
    """

    @classmethod
    def _calculate_losses(cls, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the losses for each prediction.

        :param predictions: The predictions of the model.
        :param labels: The labels of the predictions.
        """

        # Ensure we're using float64 for numerical stability
        # predictions = predictions.to(dtype=torch.float64)
        opredictions = predictions
        # Be exceptionally careful.
        # Numerically stable everything, as described in the paper.
        predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=3, keepdims=True)

        COUNT = predictions.shape[0]
        # Select the true class predictions
        y_true = predictions[np.arange(COUNT), :, :, labels[:COUNT]]
        mean_acc = np.mean(predictions[:, 0, 0, :].argmax(1) == labels[:COUNT])

        losses = np.exp(-y_true)

        return losses, mean_acc

    @classmethod
    def get_signal(cls, model, dataloader, device):
        """
        wrapper to call get_signal_reference from ReferenceModelAccess
        """
        model_access = RMIAModelAccess(model, model, ModelAccessType.BLACK_BOX)
        return model_access.get_signal_rmia(dataloader, device)

    @classmethod
    def process_shadow_models(cls, info: RMIAAuxiliaryInfo, auxiliary_dataset: Dataset, shadow_model_arch) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load and process the shadow models to generate the scores and kept indices.

        :param info: The auxiliary info instance containing all the necessary information.
        :param auxiliary_dataset: The auxiliary dataset.
        :param shadow_model_arch: The architecture of the shadow model.

        :return: The list of scores and the list of kept indices.
        """
        fullsetloader = DataLoader(auxiliary_dataset, batch_size=20, shuffle=False, num_workers=2)

        _, fullset_targets = get_xy_from_dataset(auxiliary_dataset)

        loss_list = []
        keep_list = []
        model_locations = sorted(os.listdir(info.shadow_path),
                                 key=lambda x: int(re.search(r'\d+', x).group()))  # os.listdir(info.shadow_path)

        for index, dir_name in enumerate(model_locations, start=1):
            seed_folder = os.path.join(info.shadow_path, dir_name)
            if os.path.isdir(seed_folder):
                model_path = os.path.join(seed_folder, "shadow.pth")
                print(f"load model [{index}/{len(model_locations)}]: {model_path}")
                model = LIRAUtil.load_model(shadow_model_arch, path=model_path).to(info.device)
                losses, mean_acc = cls._calculate_losses(cls.get_signal(model,
                                                                        fullsetloader,
                                                                        info.device).cpu().numpy(),
                                                         fullset_targets)
                print("Mean acc", mean_acc)
                # Convert the numpy array to a PyTorch tensor and add a new dimension
                losses = torch.unsqueeze(torch.from_numpy(losses), 0)
                loss_list.append(losses)

                keep_path = os.path.join(seed_folder, "keep.npy")
                if os.path.isfile(keep_path):
                    keep = torch.unsqueeze(torch.from_numpy(np.load(keep_path)), 0)
                    keep_list.append(keep)
            else:
                print(f"model {index} at {model_path} does not exist, skip this record")

        return loss_list, keep_list

    @classmethod
    def process_target_model(cls, target_model_access: RMIAModelAccess, info: RMIAAuxiliaryInfo,
                             dataset: Dataset) -> List[torch.Tensor]:
        """
        Calculates the target model's losses.

        :param target_model_access: The model access instance for the target model.
        :param info: The auxiliary info instance containing all the necessary information.
        :param dataset: The dataset to obtain the losses with.

        :return: The list of losses(predictive probabilities)
        """
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=8)

        _, fullset_targets = get_xy_from_dataset(dataset)

        loss_list = []

        print(f"processing target model")
        target_model_access.to_device(info.device)
        losses, mean_acc = cls._calculate_losses(
            target_model_access.get_signal_lira(dataset_loader, info.device, augmentation=18).cpu().numpy(),
            fullset_targets)

        # Convert the numpy array to a PyTorch tensor and add a new dimension
        losses = torch.unsqueeze(torch.from_numpy(losses), 0)
        loss_list.append(losses)

        return loss_list

    @classmethod
    def majority_voting_tensor(cls, tensor, axis):
        """
        Perform majority voting on a tensor along a given axis. Adapted from RMIA
        """
        return torch.mode(torch.stack(tensor), axis).values * 1.0

    @classmethod
    def RMIA_mia(cls,
                 target_signal,
                 target_indices: np.ndarray,
                 reference_keep_matrix,
                 reference_signals,
                 population_indices: np.ndarray,
                 aux_info: RMIAAuxiliaryInfo = None
                 ):
        """
        This function implements core logic of RMIA attack. It's adapted from aggregate_signals in RMIA's main.py.

        Function that aggregate signals for a given signal name under the lira setting. Returns the list of corresponding
        preds and answers.

        :param target_signal: signal matrix for a given set of target_signals of shape queried_signals
        :param target_indices: array of indices of the columns in 1, len(queried_signals) for which to compute membership
        :param reference_keep_matrix: shape (nb_ref_models-1) x queried_signals
        :param reference_signals: shape (nb_ref_models-1) x queried_signals x nb augmentations
        :param population_indices: array of indices of the columns in len(queried_signals) for which to compute population signal

        :return: prediction: the computed of target signals (preferably, all membership signals should be of
        order lower=member), answers: the membership of target signals
        """

        dat_in = []
        dat_out = []

        in_size, out_size = 100000, 100000

        for j in range(reference_signals.shape[1]):
            dat_in_j = reference_signals[reference_keep_matrix[:, j], j, :]
            dat_out_j = reference_signals[~reference_keep_matrix[:, j], j, :]

            dat_in.append(dat_in_j)
            dat_out.append(dat_out_j)

        in_size = min(min(map(len, dat_in)), in_size)
        out_size = min(min(map(len, dat_out)), out_size)

        dat_in = np.array([x[:in_size] for x in dat_in])
        dat_out = np.array([x[:out_size] for x in dat_out])

        in_signals = torch.stack([torch.from_numpy(x) for x in dat_in])  # shape dataset_size x out_or_in_size x nb_augmentations
        out_signals = torch.stack([torch.from_numpy(x) for x in dat_out])  # shape dataset_size x out_or_in_size x nb_augmentations

        ref_signal = (torch.cat((in_signals, out_signals), dim=1)).transpose(0, 1)

        all_mean_x = np.mean(ref_signal[:, target_indices, :], axis=0)
        all_mean_z = np.mean(ref_signal[:, population_indices, :], axis=0)

        augmented_gammas = []
        for k in tqdm(range(0, aux_info.nb_augmentations), desc=f"RMIA attack for each query..."):
            mean_x = all_mean_x[:, k]
            mean_z = all_mean_z[:, k]

            prob_ratio_x = (target_signal[target_indices, k].ravel() / (mean_x))
            prob_ratio_z_rev = 1 / (target_signal[population_indices, k].ravel() / (mean_z))

            # shape nb_targets x nb_population
            score = torch.outer(prob_ratio_x, prob_ratio_z_rev)
            augmented_gammas.append((score > float(aux_info.gamma)))

        augmented_test = cls.majority_voting_tensor(augmented_gammas, 0)
        del augmented_gammas

        prediction = -np.array(augmented_test.mean(1).reshape(1, len(mean_x)))
        prediction[:, population_indices] = ((prediction[:, population_indices] * len(mean_z)) - 1.0) / (
                    len(mean_z) - 1)

        return prediction

class RMIAAttack(MiAttack):
    """
    Implementation of MiAttack for RMIA.
    """

    def __init__(self, target_model_access: RMIAModelAccess, auxiliary_info: RMIAAuxiliaryInfo):
        """
        Initialize LiraAttack.
        """
        super().__init__(target_model_access, auxiliary_info)
        self.auxiliary_dataset = None
        self.shadow_scores, self.shadow_keeps = None, None
        self.auxiliary_info = auxiliary_info
        self.config = self.auxiliary_info.config
        self.target_model_access = target_model_access

    def prepare(self, auxiliary_dataset):
        """
        Prepare the attack with the auxiliary dataset. If it's an online attack, then no need to prepare.

        :param auxiliary_dataset: The auxiliary dataset to be used for the attack.
        """
        self.auxiliary_dataset = auxiliary_dataset

        # create directories
        for dir in [self.auxiliary_info.save_path, self.auxiliary_info.shadow_path, self.auxiliary_info.log_path]:
            if dir is not None:
                os.makedirs(dir, exist_ok=True)
        if self.auxiliary_info.online is False:
            raise NotImplementedError("RMIA does not support offline training yet.")
        self.prepared = True

    def infer(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        """
        Infers whether a data point is in the training set by using the Reference Attack (Attack-R).

        :param dataset: The target data points to be inferred.
        :return: The inferred membership status of the data point.
        """
        TEST = False  # if True, we save scores and keep to the file

        shadow_model = self.target_model_access.get_untrained_model()
        # concatenate the target dataset and the auxiliary dataset
        shadow_target_concat_set = ConcatDataset([self.auxiliary_dataset, dataset])
        LIRAUtil.train_shadow_models(shadow_model, shadow_target_concat_set, info=self.auxiliary_info)

        # given the model, calculate the score and generate the kept index data

        if TEST:
            # if we find the scores and keep from the file, we don't need to calculate it again
            if os.path.exists('shadow_losses.npy') and os.path.exists('shadow_keeps.npy'):
                self.shadow_losses = torch.from_numpy(np.load('shadow_losses.npy'))
                self.shadow_keeps = torch.from_numpy(np.load('shadow_keeps.npy'))
            else:
                self.shadow_losses, self.shadow_keeps = RMIAUtil.process_shadow_models(self.auxiliary_info,
                                                                                       shadow_target_concat_set,
                                                                                       shadow_model)
                # Convert the list of tensors to a single tensor
                self.shadow_scores = torch.cat(self.shadow_losses, dim=0)
                self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)
                np.save('shadow_lossess.npy', self.shadow_scores)

                # save it as txt for debugging
                # np.savetxt('shadow_scores.txt', self.shadow_scores.numpy())
                np.save('shadow_keeps.npy', self.shadow_keeps)
        else:
            self.shadow_losses, self.shadow_keeps = RMIAUtil.process_shadow_models(self.auxiliary_info,
                                                                                   shadow_target_concat_set,
                                                                                   shadow_model)
            # Convert the list of tensors to a single tensor
            self.shadow_losses = torch.cat(self.shadow_losses, dim=0)
            self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)

        # obtaining target_score, which is the prediction of the target model
        target_signal = RMIAUtil.process_target_model(self.target_model_access, self.auxiliary_info,
                                                      shadow_target_concat_set)
        target_signal = torch.cat(target_signal, dim=0)

        # get the population indices
        target_indices = np.concatenate([np.zeros(len(self.auxiliary_dataset), np.ones(len(dataset)))], dtype=bool)
        population_indices = ~target_indices
        reference_keep_matrix = self.shadow_keeps.numpy()

        predictions = RMIAUtil.RMIA_mia(target_signal, target_indices, reference_keep_matrix, self.shadow_scores,
                                         population_indices, self.auxiliary_info)

        print(f"prediction shape: {(-predictions[-len(dataset):]).shape}")

        # return the predictions on the target data
        return -predictions[-len(dataset):]
