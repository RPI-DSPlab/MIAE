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

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils
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
        self.gamma = config.get('gamma', 2.0)
        self.query_batch_size = config.get('query_batch_size', 128)
        self.proportiontocut = config.get('proportiontocut', 0.0)  # proportion to cut for trim_mean
        self.taylor_n = config.get('taylor_n', 4)
        self.taylor_m = config.get('taylor_m', 0.6)
        self.temperature = config.get('temperature', 2.0) # temperature for softmax
        self.signal_metric = config.get('signal_metric', 'softmax') # softmax, taylor, soft-margin, taylor-soft-margin

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)

        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.log_path is not None:
            self.logger = logging.getLogger('rmia_logger')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/rmia.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


def _split_data(fullset, expid, iteration_range):
    keep = np.random.uniform(0, 1, size=(iteration_range, len(fullset)))
    order = keep.argsort(0)
    keep = order < int(.5 * iteration_range)
    keep = np.array(keep[expid], dtype=bool)
    return np.where(keep)[0], np.where(~keep)[0]


class RMIAUtil(MIAUtils):
    """
    RMIA shares many of the code with LIRA attack, so we only define methods unique to RMIA here.
    """

    @classmethod
    def convert_signal(cls, all_logits: torch.tensor, all_true_labels: torch.tensor, 
                       metric: str, temp: float, extra=None):
        """
        Convert the logits to signals based on the metric. Adapted from RMIA code base.
        
        param: all_logits: The logits of the model.
        param: all_true_labels: The true labels of the model.
        param: metric: The metric to use for the conversion.
        param: temp: The temperature to use for the conversion.
        param: extra: Extra parameters for the conversion, ie: taylor_n, taylor_m

        return: The converted signals.
        """
        def factorial(n):
            fact = 1
            for i in range(2, n + 1):
                fact = fact * i
            return fact
        def get_taylor(logit_signals, n):
            power = logit_signals
            taylor = power + 1.0
            for i in range(2, n):
                power = power * logit_signals
                taylor = taylor + (power / factorial(i))
            return taylor
        
        all_logits = all_logits.cpu()
        all_true_labels = all_true_labels.cpu()

        if metric == 'softmax':
            logit_signals = torch.div(all_logits, temp)
            max_logit_signals, max_indices = torch.max(logit_signals, dim=1)
            logit_signals = torch.sub(logit_signals, max_logit_signals.reshape(-1, 1))
            exp_logit_signals = torch.exp(logit_signals)
            exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
            true_exp_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
            output_signals = torch.div(true_exp_logit, exp_logit_sum)
        elif metric == 'taylor':
            n = extra["taylor_n"]
            taylor_signals = get_taylor(all_logits, n)
            taylor_logit_sum = taylor_signals.sum(axis=1).reshape(-1, 1)
            true_taylor_logit = taylor_signals.gather(1, all_true_labels.reshape(-1, 1))
            output_signals = torch.div(true_taylor_logit, taylor_logit_sum)
        elif metric == 'soft-margin':
            m = float(extra["taylor_m"])
            logit_signals = torch.div(all_logits, temp)
            exp_logit_signals = torch.exp(logit_signals)
            exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
            true_logits = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
            exp_true_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
            exp_logit_sum = exp_logit_sum - exp_true_logit
            soft_true_logit = torch.exp(true_logits - m)
            exp_logit_sum = exp_logit_sum + soft_true_logit
            output_signals = torch.div(soft_true_logit, exp_logit_sum)
        elif metric == 'taylor-soft-margin':
            m, n = float(extra["taylor_m"]), int(extra["taylor_n"])
            logit_signals = torch.div(all_logits, temp)
            taylor_logits = get_taylor(logit_signals, n)
            taylor_logit_sum = taylor_logits.sum(axis=1).reshape(-1, 1)
            true_logit = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
            taylor_true_logit = taylor_logits.gather(1, all_true_labels.reshape(-1, 1))
            taylor_logit_sum = taylor_logit_sum - taylor_true_logit
            soft_taylor_true_logit = get_taylor(true_logit - m, n)
            taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
            output_signals = torch.div(soft_taylor_true_logit, taylor_logit_sum)
        return torch.flatten(output_signals)
    
    @classmethod
    def convert_signal_wrapper(cls, all_logits: torch.tensor, all_true_labels: torch.tensor, info: RMIAAuxiliaryInfo):
        """
        Wrapper to call convert_signal from RMIAUtil.

        :param all_logits: The logits of the model.
        :param all_true_labels: The true labels of the model.
        :param info: The auxiliary info instance containing all the necessary information.

        :return: The converted signals.
        """
        temp = info.temperature
        if info.nb_augmentations == 1:
            if info.signal_metric == 'softmax':
                return cls.convert_signal(all_logits, all_true_labels, info.signal_metric, temp)
            elif info.signal_metric == 'taylor':
                return cls.convert_signal(all_logits, all_true_labels, info.signal_metric, temp, extra={"taylor_n": info.taylor_n})
            elif info.signal_metric == 'soft-margin':
                return cls.convert_signal(all_logits, all_true_labels, info.signal_metric, temp, extra={"taylor_m": info.taylor_m})
            elif info.signal_metric == 'taylor-soft-margin':
                return cls.convert_signal(all_logits, all_true_labels, info.signal_metric, temp, extra={"taylor_m": info.taylor_m, "taylor_n": info.taylor_n})
            else:
                raise ValueError(f"Metric {info.signal_metric} is not supported, and it was not implemented in the RMIA paper.")
        else:
            logits_convert = []
            all_logits = all_logits.squeeze(1)
            for aug_index in range(all_logits.shape[1]):
                if info.signal_metric == 'softmax':
                    logits_convert.append(cls.convert_signal(all_logits[:, aug_index, :], all_true_labels, info.signal_metric, temp))
                elif info.signal_metric == 'taylor':
                    logits_convert.append(cls.convert_signal(all_logits[:, aug_index, :], all_true_labels, info.signal_metric, temp, extra={"taylor_n": info.taylor_n}))
                elif info.signal_metric == 'soft-margin':
                    logits_convert.append(cls.convert_signal(all_logits[:, aug_index, :], all_true_labels, info.signal_metric, temp, extra={"taylor_m": info.taylor_m}))
                elif info.signal_metric == 'taylor-soft-margin':
                    logits_convert.append(cls.convert_signal(all_logits[:, aug_index, :], all_true_labels, info.signal_metric, temp, extra={"taylor_m": info.taylor_m, "taylor_n": info.taylor_n}))
                else:
                    raise ValueError(f"Metric {info.signal_metric} is not supported, and it was not implemented in the RMIA paper.")
            # convert the list of tensors to a single tensor and restore its shape
            converted_signal = torch.stack(logits_convert, dim=1)
            return converted_signal
    @classmethod
    def get_signal(cls, model, dataloader, device):
        """
        wrapper to call get_signal_lira from ReferenceModelAccess.
        """
        model_access = RMIAModelAccess(model, model, ModelAccessType.BLACK_BOX)
        model_access.to_device(device)
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
        fullsetloader = DataLoader(auxiliary_dataset, batch_size=info.query_batch_size, shuffle=False, num_workers=2)

        _, fullset_targets = get_xy_from_dataset(auxiliary_dataset)

        signal_list = []
        keep_list = []
        model_locations = sorted(os.listdir(info.shadow_path),
                                 key=lambda x: int(re.search(r'\d+', x).group()))  # os.listdir(info.shadow_path)

        for index, dir_name in tqdm(enumerate(model_locations, start=1), desc="Processing shadow models"):
            seed_folder = os.path.join(info.shadow_path, dir_name)
            if os.path.isdir(seed_folder):
                model_path = os.path.join(seed_folder, "shadow.pth")
                cls.log(info, f"load model [{index}/{len(model_locations)}]: {model_path}", print_flag=False)
                model = LIRAUtil.load_model(shadow_model_arch, path=model_path).to(info.device)
                
                raw_signal = cls.get_signal(model, fullsetloader, info.device)
                # convert the raw signal to RMIA's required signal
                targets = torch.tensor(fullset_targets)
                converted_signal = cls.convert_signal_wrapper(raw_signal, targets, info)
                signal_list.append(converted_signal.unsqueeze(0))

                keep_path = os.path.join(seed_folder, "keep.npy")
                if os.path.isfile(keep_path):
                    keep = torch.unsqueeze(torch.from_numpy(np.load(keep_path)), 0)
                    keep_list.append(keep)
            else:
                cls.log(info, f"model {index} at {model_path} does not exist, skip this record", print_flag=True)

        return signal_list, keep_list

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
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=info.query_batch_size, shuffle=False, num_workers=8)

        _, fullset_targets = get_xy_from_dataset(dataset)

        signal_list = []

        cls.log(info, f"processing target model", print_flag=True)
        target_model_access.to_device(info.device)
        raw_signal = target_model_access.get_signal_lira(dataset_loader, info.device, augmentation=18)
        # convert the raw signal to RMIA's required signal
        targets = torch.tensor(fullset_targets)
        converted_signal = cls.convert_signal_wrapper(raw_signal, targets, info)
        signal_list.append(converted_signal.unsqueeze(0))

        return signal_list

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

        :param target_signal (dataset_size, num_aug): signal matrix for a given set of target_signals of shape queried_signals
        :param target_indices: array of indices of the columns in 1, len(queried_signals) for which to compute membership
        :param reference_keep_matrix: shape (nb_ref_models-1) x queried_signals
        :param reference_signals: shape (nb_ref_models-1) x queried_signals x nb augmentations
        :param population_indices: array of indices of the columns in len(queried_signals) for which to compute population signal

        :return: prediction: the computed of target signals (preferably, all membership signals should be of
        order lower=member), answers: the membership of target signals
        """

        in_signals = []
        out_signals = []


        for j in range(reference_signals.shape[1]):
            dat_in_j = reference_signals[reference_keep_matrix[:, j], j, :]
            dat_out_j = reference_signals[~reference_keep_matrix[:, j], j, :]

            in_signals.append(dat_in_j)
            out_signals.append(dat_out_j)

        in_size = min(map(len, in_signals))
        out_size = min(map(len, out_signals))
        out_or_in_size = min(in_size, out_size)
        print(f"out_or_in_size: {out_or_in_size}")

        in_signals = torch.stack([x[:out_or_in_size] for x in in_signals])
        out_signals = torch.stack([x[:out_or_in_size] for x in out_signals])

        ref_signals = (torch.cat((in_signals, out_signals), dim=1)).transpose(0, 1)

        all_mean_x = trim_mean(ref_signals[:, target_indices, :], proportiontocut=aux_info.proportiontocut, axis=0)
        all_mean_z = trim_mean(ref_signals[:, population_indices, ], proportiontocut=aux_info.proportiontocut, axis=0)

        augmented_gammas = []
        for k in range(0, aux_info.nb_augmentations):
            mean_x = all_mean_x[:, k]
            mean_z = all_mean_z[:, k]
            target_signal = target_signal.squeeze(dim=0)

            prob_ratio_x = (target_signal[target_indices, k].ravel() / (mean_x))
            prob_ratio_z_rev = 1 / (target_signal[population_indices, k].ravel() / (mean_z))

            # shape nb_targets x nb_population
            score = torch.outer(prob_ratio_x, prob_ratio_z_rev)
            augmented_gammas.append((score > float(aux_info.gamma)))

        augmented_test = cls.majority_voting_tensor(augmented_gammas, 0)
        del augmented_gammas

        prediction = -np.array(augmented_test.mean(1).reshape(1, len(mean_x))).transpose()
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
        self.shadow_signals, self.shadow_keeps = None, None
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
        TEST = True  # if True, we save scores and keep to the file

        shadow_model = self.target_model_access.get_untrained_model()
        # concatenate the target dataset and the auxiliary dataset
        shadow_target_concat_set = ConcatDataset([self.auxiliary_dataset, dataset])
        LIRAUtil.train_shadow_models(shadow_model, shadow_target_concat_set, info=self.auxiliary_info)

        # given the model, calculate the score and generate the kept index data

        if TEST:
            # if we find the scores and keep from the file, we don't need to calculate it again
            if os.path.exists('shadow_signals_rmia.npy') and os.path.exists('shadow_keeps_rmia.npy'):
                self.shadow_signals = torch.from_numpy(np.load('shadow_signals_rmia.npy'))
                self.shadow_keeps = torch.from_numpy(np.load('shadow_keeps_rmia.npy'))
            else:
                self.shadow_signals, self.shadow_keeps = RMIAUtil.process_shadow_models(self.auxiliary_info,
                                                                                       shadow_target_concat_set,
                                                                                       shadow_model)
                # Convert the list of tensors to a single tensor
                self.shadow_signals = torch.cat(self.shadow_signals, dim=0)
                self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)
                np.save('shadow_signals_rmia.npy', self.shadow_signals)

                # save it as txt for debugging
                np.save('shadow_keeps_rmia.npy', self.shadow_keeps)
        else:
            self.shadow_signals, self.shadow_keeps = RMIAUtil.process_shadow_models(self.auxiliary_info,
                                                                                   shadow_target_concat_set,
                                                                                   shadow_model)
            # Convert the list of tensors to a single tensor
            self.shadow_signals = torch.cat(self.shadow_signals, dim=0)
            self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)

        # obtaining target_score, which is the prediction of the target model
        target_signals = RMIAUtil.process_target_model(self.target_model_access, self.auxiliary_info, shadow_target_concat_set)
        target_signals = torch.cat(target_signals, dim=0)

        # get the population indices
        target_indices = np.concatenate([np.zeros(len(self.auxiliary_dataset)).astype(bool), np.ones(len(dataset)).astype(bool)])
        population_indices = ~target_indices
        reference_keep_matrix = self.shadow_keeps.numpy()
        predictions = RMIAUtil.RMIA_mia(target_signals, target_indices, reference_keep_matrix, self.shadow_signals,
                                         population_indices, self.auxiliary_info)



        # return the predictions on the target data
        return -predictions[-len(dataset):]
