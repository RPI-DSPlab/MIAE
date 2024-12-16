# This code implements "Enhanced Membership Inference Attacks against Machine Learning Models" by Ye et al.
# The code is based on the code from
# https://github.com/privacytrustlab/ml_privacy_meter/tree/295e7e37e889e12df4083b812f71ed2e2ddd8b4a/research/2022_enhanced_mia
# Since Ye et al. implemented their Attack-R based on the code for LIRA attack, we will be reusing
# many components from the LIRA attack code.

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

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils
from miae.utils.dataset_utils import get_xy_from_dataset

from miae.attacks.lira_mia import LIRAUtil


class ReferenceModelAccess(ModelAccess):
    """
    Implementation of ModelAccess for Reference Attack (Attack-R from Enhanced MIA paper).
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType = ModelAccessType.BLACK_BOX):
        """
        Initialize LiraModelAccess.
        """
        super().__init__(model, untrained_model, access_type)
        self.model = model
        self.model.eval()

    def get_signal_reference(self, dataloader, device):
        """
        Generates logits for a dataloader given a model

        Args:
        model (torch.nn.Module): The PyTorch model to generate logits.
        data: a data point
        device: the device (cpu or cuda) where the computations will take place.
        """

        self.model.eval()
        all_logits = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                outputs = self.model(images)
                all_logits.append(outputs.unsqueeze(1))
                
        all_logits = torch.cat(all_logits, dim=0)
        all_logits = all_logits.unsqueeze(1)
        return all_logits


class ReferenceAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of AuxiliaryInfo for Attack R.
    """

    def __init__(self, config):
        """
        Initialize ReferenceAuxiliaryInfo with a configuration dictionary.
        """
        super().__init__(config)
        self.config = config

        # Training parameters
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.lr = config.get('lr', 0.1)
        self.momentum = config.get('momentum', 0.9)
        self.decay = config.get('decay', 0.9999)
        self.seed = config.get('seed', 24)
        self.epochs = config.get('epochs', 100)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('shadow_batchsize', 128)

        # Model saving and loading parameters
        self.save_path = config.get('save_path', None)

        # Auxiliary info for reference attack
        self.num_shadow_models = config.get('num_shadow_models', 29)  # paper default is 29
        self.shadow_path = config.get('shadow_path', f"{self.save_path}/weights/shadow/")
        self.query_batch_size = config.get('query_batch_size', 512)
        self.shadow_diff_init = config.get('shadow_diff_init', False)

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)

        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.log_path is not None:
            self.logger = logging.getLogger('reference_logger')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/reference.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


def _split_data(fullset, expid, iteration_range):
    keep = np.random.uniform(0, 1, size=(iteration_range, len(fullset)))
    order = keep.argsort(0)
    keep = order < int(.5 * iteration_range)
    keep = np.array(keep[expid], dtype=bool)
    return np.where(keep)[0], np.where(~keep)[0]


class ReferenceUtil(MIAUtils):
    """
    Attack-R shares most of the code with LIRA attack, so we only define methods unique to Attack-R here.
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
        model_access = ReferenceModelAccess(model, model, ModelAccessType.BLACK_BOX)
        return model_access.get_signal_reference(dataloader, device)

    @classmethod
    def process_shadow_models(cls, info: ReferenceAuxiliaryInfo, auxiliary_dataset: Dataset, shadow_model_arch) \
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
                cls.log(info, f"load model [{index}/{len(model_locations)}]: {model_path}", print_flag=True)
                model = LIRAUtil.load_model(shadow_model_arch, path=model_path).to(info.device)
                losses, mean_acc = cls._calculate_losses(cls.get_signal(model,
                                                                        fullsetloader,
                                                                        info.device).cpu().numpy(),
                                                         fullset_targets)
                cls.log(info, f"mean acc: {mean_acc}", print_flag=True)
                # Convert the numpy array to a PyTorch tensor and add a new dimension
                losses = torch.unsqueeze(torch.from_numpy(losses), 0)
                loss_list.append(losses)

                keep_path = os.path.join(seed_folder, "keep.npy")
                if os.path.isfile(keep_path):
                    keep = torch.unsqueeze(torch.from_numpy(np.load(keep_path)), 0)
                    keep_list.append(keep)
            else:
                cls.log(info, f"model {index} at {model_path} does not exist, skip this record", print_flag=True)

        return loss_list, keep_list

    @classmethod
    def process_target_model(cls, target_model_access: ReferenceModelAccess, info: ReferenceAuxiliaryInfo,
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

        cls.log(info, f"processing target model", print_flag=True)
        target_model_access.to_device(info.device)
        losses, mean_acc = cls._calculate_losses(
            target_model_access.get_signal_reference(dataset_loader, info.device).cpu().numpy(), fullset_targets)

        # Convert the numpy array to a PyTorch tensor and add a new dimension
        losses = torch.unsqueeze(torch.from_numpy(losses), 0)
        loss_list.append(losses)

        return loss_list

    @classmethod
    def reference_mia(cls, losses, check_losses):
        """
        Implements the core logic of the Attack-R method from the Enhanced MIA paper.

        :param losses: The losses of the shadow models.
        :param check_losses: The losses of the target model.
        """

        dat_reference = np.log(np.exp(-losses) / (1 - np.exp(-losses))).numpy()
        mean_reference = np.mean(dat_reference, 0)
        std_reference = np.std(dat_reference, 0)
        check_losses = np.transpose(check_losses[0])
        check_losses = np.transpose(check_losses, (2, 1, 0))
        prediction = 1 - scipy.stats.norm.cdf(np.log(np.exp(-check_losses) / (1 - np.exp(-check_losses))),
                                              mean_reference, std_reference + 1e-30)
        return np.array(prediction).reshape(-1, 1).squeeze()


class ReferenceAttack(MiAttack):
    """
    Implementation of MiAttack for Reference Attack (Attack-R from Enhanced MIA paper).
    """

    def __init__(self, target_model_access: ReferenceModelAccess, auxiliary_info: ReferenceAuxiliaryInfo):
        """
        Initialize Attack-R.
        """
        super().__init__(target_model_access, auxiliary_info)
        self.auxiliary_dataset = None
        self.shadow_scores, self.shadow_keeps = None, None
        self.aux_info = auxiliary_info
        self.config = self.aux_info.config
        self.target_model_access = target_model_access

    def prepare(self, auxiliary_dataset):
        """
        Since Reference attack trains shadow models with/without the target dataset, we don't need to prepare
        anything here.

        :param auxiliary_dataset: The auxiliary dataset to be used for the attack.
        """
        self.auxiliary_dataset = auxiliary_dataset

        # create directories
        for dir in [self.aux_info.save_path, self.aux_info.shadow_path, self.aux_info.log_path]:
            if dir is not None:
                os.makedirs(dir, exist_ok=True)

        ReferenceUtil.log(self.aux_info, "Start preparing the attack...", print_flag=True)
        ReferenceUtil.log(self.aux_info, "Finish preparing the attack...", print_flag=True)
        self.prepared = True
        

    def infer(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        """
        Infers whether a data point is in the training set by using the Reference Attack (Attack-R).

        :param dataset: The target data points to be inferred.
        :return: The inferred membership status of the data point.
        """
        TEST = False  # if True, we save scores and keep to the file

        ReferenceUtil.log(self.aux_info, "Start membership inference...", print_flag=True)

        shadow_model = self.target_model_access.get_untrained_model()
        # concatenate the target dataset and the auxiliary dataset
        shadow_target_concat_set = ConcatDataset([self.auxiliary_dataset, dataset])
        LIRAUtil.train_shadow_models(shadow_model, shadow_target_concat_set, info=self.aux_info)

        # given the model, calculate the score and generate the kept index data

        if TEST:
            # if we find the scores and keep from the file, we don't need to calculate it again
            if os.path.exists('shadow_losses.npy') and os.path.exists('shadow_keeps.npy'):
                self.shadow_losses = torch.from_numpy(np.load('shadow_losses.npy'))
                self.shadow_keeps = torch.from_numpy(np.load('shadow_keeps.npy'))
            else:
                self.shadow_losses, self.shadow_keeps = ReferenceUtil.process_shadow_models(self.aux_info,
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
            self.shadow_losses, self.shadow_keeps = ReferenceUtil.process_shadow_models(self.aux_info,
                                                                                        shadow_target_concat_set,
                                                                                        shadow_model)
            # Convert the list of tensors to a single tensor
            self.shadow_losses = torch.cat(self.shadow_losses, dim=0)
            self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)

        # obtaining target_score, which is the prediction of the target model
        target_losses = ReferenceUtil.process_target_model(self.target_model_access, self.aux_info,
                                                      shadow_target_concat_set)
        target_losses = torch.cat(target_losses, dim=0)

        predictions = ReferenceUtil.reference_mia(self.shadow_losses, target_losses)


        ReferenceUtil.log(self.aux_info, "Finish membership inference...", print_flag=True)
        # return the predictions on the target data
        return -predictions[-len(dataset):]
