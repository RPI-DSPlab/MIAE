# This code implements "Membership Inference Attacks From First Principles", S&P 2022
# The code is based on the code from
# https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021
import copy
import logging
import os
import re
from typing import List, Tuple

import numpy as np
import scipy
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torch.nn as nn
from tqdm import tqdm

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils
from miae.utils.dataset_utils import get_xy_from_dataset
from miae.utils.set_seed import set_seed


class LiraModelAccess(ModelAccess):
    """
    Implementation of ModelAccess for Lira.
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType = ModelAccessType.BLACK_BOX):
        """
        Initialize LiraModelAccess.
        """
        super().__init__(model, untrained_model, access_type)
        self.model = model
        self.model.eval()


class LiraAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of AuxiliaryInfo for Lira.
    """

    def __init__(self, config):
        """
        Initialize LiraAuxiliaryInfo with a configuration dictionary.
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

        # Auxiliary info for LIRA
        self.num_shadow_models = config.get('num_shadow_models', 20)
        self.shadow_path = config.get('shadow_path', f"{self.save_path}/weights/shadow/")
        self.online = config.get('online', True)
        self.fix_variance = config.get('fix_variance', True)
        self.query_batch_size = config.get('query_batch_size', 256)
        self.shadow_diff_init = config.get('shadow_diff_init', False) # whether to re-init every shadow model
        self.augmentation_query = config.get('augmentation_query', 18)

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)

        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.log_path is not None:
            self.logger = logging.getLogger('lira_logger')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/lira.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


def _split_data(fullset, expid, iteration_range, seed):
    np.random.seed(seed) # make sure the seed is set correct
    keep = np.random.uniform(0, 1, size=(iteration_range, len(fullset)))
    order = keep.argsort(0)
    keep = order < int(.5 * iteration_range)
    keep = np.array(keep[expid], dtype=bool)
    return np.where(keep)[0], np.where(~keep)[0]


class LIRAUtil(MIAUtils):
    @classmethod
    def _make_directory_if_not_exists(cls, dir_path):
        """
        Checks if a directory exists and, if not, creates it.

        Args:
        dir_path (str): The path of the directory to be checked/created.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            return False
        return True

    @classmethod
    def train(cls, model, device, train_loader, optimizer, scheduler=None):
        """
        train function for the shadow_model
        """
        model.train()
        # ema = EMA(model, 0.999)

        running_loss = 0.0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # calculate running loss and correct prediction count for accuracy
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        if scheduler is not None:
            scheduler.step()

        return running_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

    @classmethod
    def test(cls, model, device, test_loader):
        """
        test function for the shadow_model
        """
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return correct / len(test_loader.dataset)

    @classmethod
    def predict(cls, model, loader, device):
        """
        Predicts the outputs of a model given a data loader.
        """
        model.eval()
        outputs = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(device)
                output = nn.Softmax(dim=1)(model(data))
                outputs.append(output)
        return torch.cat(outputs)

    @classmethod
    def save_model(cls, model, path):
        """
        Saves the model to the given path.
        """
        torch.save(model.state_dict(), path)

    @classmethod
    def load_model(cls, model, path):
        """
        Loads the model from the given path.
        """
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    @classmethod
    def train_shadow_models(cls, model, dataset, info: LiraAuxiliaryInfo):
        """
        Trains the shadow models.
        :param model: Untrained copy of the target model.
        :param dataset: The dataset to be used for training.
        :param info: The auxiliary info instance containing all the necessary information.
        """
        # init
        set_seed(info.seed)
        iteration_range = info.num_shadow_models
        device = torch.device(info.device)

        if not os.path.exists(info.shadow_path):
            os.makedirs(info.shadow_path)

        # if the required shadow models are already trained, skip the training
        if len(os.listdir(info.shadow_path)) >= iteration_range:
            cls.log(info, f"shadow models are already trained, skip the training", print_flag=True)
            return


        for expid in range(iteration_range):
            # Define the directory path
            folder_name = expid
            dir_path = f"{info.shadow_path}/{folder_name}"

            if info.shadow_diff_init:
                try:
                    set_seed((info.seed + expid)*100) # *100 to avoid overlapping of different instances
                    model.initialize_weights()
                except:
                    raise NotImplementedError("the model doesn't have .initialize_weights method")
                
            set_seed(info.seed)

            # Check if the directory exists and create
            if os.path.exists(dir_path):
                if os.path.exists(dir_path + "/shadow.pth") and os.path.exists(dir_path + "/keep.npy"):
                    cls.log(info, f"shadow model {expid} already exists at {dir_path}, skip training", print_flag=True)
                    continue
            else:
                cls._make_directory_if_not_exists(dir_path)

            # split the data
            shadow_train_indices, shadow_out_indices = _split_data(dataset, expid, iteration_range, info.seed)

            # Create the data loaders for training and testing
            shadow_train_loader = DataLoader(Subset(dataset, shadow_train_indices), batch_size=info.batch_size,
                                             shuffle=True)
            shadow_out_loader = DataLoader(Subset(dataset, shadow_out_indices), batch_size=info.batch_size,
                                           shuffle=False)

            curr_model = copy.deepcopy(model)
            curr_model.to(device)

            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, curr_model.parameters()),
                                        lr=info.lr, momentum=info.momentum, weight_decay=info.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, info.epochs)

            cls.log(info, f"training shadow model #{expid} with "
                          f"train size: {len(shadow_train_indices)} and test size: {len(shadow_out_indices)}",
                    print_flag=True)

            for epoch in tqdm(range(1, info.epochs + 1)):
                # print the length of the train and test set
                loss, train_acc = LIRAUtil.train(curr_model, device, shadow_train_loader, optimizer,
                                                 scheduler=scheduler)
                test_acc = LIRAUtil.test(curr_model, device, shadow_out_loader)
                if (epoch % 20 == 0 or epoch == info.epochs):
                    cls.log(info, f"Train Shadow Model #{expid}: {epoch}/{info.epochs}: TRAIN loss: {loss:.3f}, "
                                    f"TRAIN acc: {train_acc * 100:.3f}%, TEST acc: {test_acc * 100:.3f}%, lr: {scheduler.get_last_lr()[0]: .4f}",
                            print_flag=True)


            # save model
            LIRAUtil.save_model(curr_model, f"{dir_path}/shadow.pth")
            # save keep
            is_in_train = np.full(len(dataset), False)
            is_in_train[shadow_train_indices] = True
            np.save(f"{dir_path}/keep.npy", is_in_train)

    @classmethod
    def lira_mia(cls, keep, scores, check_scores, in_size=100000, out_size=100000,
                 fix_variance=True):
        """
        Implements the core logic of the LIRA membership inference attack.

        Args:
        keep (np.ndarray): An array indicating In samples.
        scores (np.ndarray): An array containing the scores of the samples.
        check_scores (np.ndarray): An array containing the scores of the samples for target model.
        in_size (int):
        out_size (int):
        fix_variance (bool): If true, the variance is fixed.
        """
        dat_in = []
        dat_out = []

        for j in range(scores.shape[1]):
            dat_in_j = scores[keep[:, j], j, :]
            dat_out_j = scores[~keep[:, j], j, :]

            dat_in.append(dat_in_j)
            dat_out.append(dat_out_j)

        in_size = min(min(map(len, dat_in)), in_size)
        out_size = min(min(map(len, dat_out)), out_size)

        dat_in = np.array([x[:in_size] for x in dat_in])
        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_in = np.median(dat_in, 1)
        mean_out = np.median(dat_out, 1)

        # Ensure no NaNs or Infs in means and stds
        mean_in = np.nan_to_num(mean_in, nan=0.0)
        mean_out = np.nan_to_num(mean_out, nan=0.0)
        if fix_variance:
            std_in = np.std(dat_in)
            std_out = np.std(dat_in)
        else:
            std_in = np.std(dat_in, 1)
            std_out = np.std(dat_out, 1)

        std_in = np.nan_to_num(std_in, nan=1.0)
        std_out = np.nan_to_num(std_out, nan=1.0)

        prediction = []

        for sc in check_scores:
            pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out

            prediction.extend(score.mean(1))

        return np.array(prediction)

    @classmethod
    def lira_mia_offline(cls, keep, scores, check_scores, in_size=100000, out_size=100000,
                        fix_variance=True):
        """
        Implements the LIRA membership inference attack in an offline setting using the scores and 
        ground truth answer from check_keep to predict whether examples in check_scores were training data.
        
        Args:
        keep (np.ndarray): An array indicating IN samples on the shadow model.
        scores (np.ndarray): An array containing the scores of the aux dataset from the shadow models
        check_scores (np.ndarray): An array containing the scores of the samples for target model.
        in_size (int): 
        out_size (int): 
        fix_variance (bool): If True, fixes the variance across all samples.
        """
        dat_out = []

        for j in range(scores.shape[1]):
            dat_out.append(scores[~keep[:, j], j, :])

        # Limit the number of samples for training and testing
        out_size = min(min(map(len, dat_out)), out_size)

        # Slice the data to fit the sizes
        dat_out = np.array([x[:out_size] for x in dat_out])

        # Calculate median for training and test data
        mean_out = np.median(dat_out, 1)

        # Fix the variance if requested
        if fix_variance:
            std_out = np.std(dat_out)
        else:
            std_out = np.std(dat_out, 1)

        std_out = np.nan_to_num(std_out, nan=1.0)


        prediction = []
        for sc in check_scores:
            score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            prediction.extend(score.mean(1))

        return np.array(prediction)

    @classmethod
    def _generate_logits(cls, model, data_loader, augmentation, device):
        """
        warpper function for get_signal_lira
        """
        model_access = LiraModelAccess(model, model)
        return model_access.get_signal_lira(data_loader, device, augmentation=augmentation)

    @classmethod
    def process_shadow_models(cls, info: LiraAuxiliaryInfo, auxiliary_dataset: Dataset, shadow_model_arch) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load and process the shadow models to generate the scores and kept indices.

        :param info: The auxiliary info instance containing all the necessary information.
        :param auxiliary_dataset: The auxiliary dataset.
        :param shadow_model_arch: The architecture of the shadow model.
        :param threshold_acc: The accuracy threshold for skipping records.

        :return: The list of scores and the list of kept indices.
        """
        fullsetloader = DataLoader(auxiliary_dataset, batch_size=info.query_batch_size, shuffle=False, num_workers=2)

        _, fullset_targets = get_xy_from_dataset(auxiliary_dataset)

        score_list = []
        keep_list = []
        model_locations = sorted(os.listdir(info.shadow_path),
                                 key=lambda x: int(re.search(r'\d+', x).group()))  # os.listdir(info.shadow_path)

        for index, dir_name in enumerate(model_locations, start=1):
            seed_folder = os.path.join(info.shadow_path, dir_name)
            if os.path.isdir(seed_folder):
                model_path = os.path.join(seed_folder, "shadow.pth")
                cls.log(info, f"load model [{index}/{len(model_locations)}]: {model_path}", print_flag=True)
                model = cls.load_model(shadow_model_arch, path=model_path).to(info.device)
                # print(shadow_model_arch, model_path)
                scores, mean_acc = cls._calculate_score(cls._generate_logits(model,
                                                                             fullsetloader,
                                                                             info.augmentation_query,
                                                                             info.device).cpu().numpy(),
                                                        fullset_targets)
                cls.log(info, f"Model {index} mean acc: {mean_acc}", print_flag=True)
                # Convert the numpy array to a PyTorch tensor and add a new dimension
                scores = torch.unsqueeze(torch.from_numpy(scores), 0)
                score_list.append(scores)

                keep_path = os.path.join(seed_folder, "keep.npy")
                if os.path.isfile(keep_path):
                    keep = torch.unsqueeze(torch.from_numpy(np.load(keep_path)), 0)
                    keep_list.append(keep)
            else:
                cls.log(info, f"model {index} at {model_path} does not exist, skip this record", print_flag=True)

        return score_list, keep_list

    @classmethod
    def process_target_model(cls, target_model_access: LiraModelAccess, info: LiraAuxiliaryInfo,
                             dataset: Dataset) -> List[torch.Tensor]:
        """
        Calculates the target model's scores.

        :param target_model_access: The model access instance for the target model.
        :param info: The auxiliary info instance containing all the necessary information.
        :param dataset: The dataset to obtain the scores with.

        :return: The list of scores(predictive probabilities)
        """
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=info.query_batch_size, shuffle=False, num_workers=8)

        _, fullset_targets = get_xy_from_dataset(dataset)

        score_list = []

        cls.log(info, f"processing target model", print_flag=True)
        target_model_access.to_device(info.device)
        scores, mean_acc = cls._calculate_score(
            target_model_access.get_signal_lira(dataset_loader, info.device, info.augmentation_query).cpu().numpy(), fullset_targets)

        # Convert the numpy array to a PyTorch tensor and add a new dimension
        scores = torch.unsqueeze(torch.from_numpy(scores), 0)
        score_list.append(scores)

        return score_list

    @classmethod
    def _calculate_score(cls, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the score for each prediction by log logit scaling

        Args:
        predictions (torch.Tensor): The tensor of model predictions.
        labels (torch.Tensor): The tensor of true labels.
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

        # Zero out the true class predictions
        predictions[np.arange(COUNT), :, :, labels[:COUNT]] = 0

        # Calculate sum of predictions for incorrect classes
        y_wrong = np.sum(predictions, axis=3)

        # Calculate log-odds of correct versus incorrect predictions
        score = (np.log(y_true.mean(axis=1) + 1e-45) - np.log(y_wrong.mean(axis=1) + 1e-45))
        return score, mean_acc


class LiraAttack(MiAttack):
    """
    Implementation of MiAttack for Lira.
    """

    def __init__(self, target_model_access: LiraModelAccess, auxiliary_info: LiraAuxiliaryInfo):
        """
        Initialize LiraAttack.
        """
        super().__init__(target_model_access, auxiliary_info)
        self.auxiliary_dataset = None
        self.shadow_scores, self.shadow_keeps = None, None
        self.aux_info = auxiliary_info
        self.config = self.aux_info.config
        self.target_model_access = target_model_access

    def prepare(self, auxiliary_dataset):
        """
        Since LIRA trains shadow models with/without the target dataset, we don't need to prepare anything here.

        :param auxiliary_dataset: The auxiliary dataset to be used for the attack.
        """
        set_seed(self.aux_info.seed)
        LIRAUtil.log(self.aux_info, "Start preparing the attack...", print_flag=True)
        self.auxiliary_dataset = auxiliary_dataset

        # create directories
        for dir in [self.aux_info.save_path, self.aux_info.shadow_path, self.aux_info.log_path]:
            if dir is not None:
                os.makedirs(dir, exist_ok=True)

        """
        the snippet below is for the disjoint case mentioned in the LiRA paper
        """

        # if self.aux_info.online is False: # lira offline
        #     shadow_model = self.target_model_access.get_untrained_model()
        #     LIRAUtil.train_shadow_models(shadow_model, self.auxiliary_dataset, info=self.aux_info)
        #     self.shadow_scores, self.shadow_keeps = LIRAUtil.process_shadow_models(self.aux_info,
        #                                                                            self.auxiliary_dataset,
        #                                                                            shadow_model)
        #     # Convert the list of tensors to a single tensor
        #     self.shadow_scores = torch.cat(self.shadow_scores, dim=0)
        #     self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)

            
        self.prepared = True
        LIRAUtil.log(self.aux_info, "Finish preparing the attack...", print_flag=True)

    def infer(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        """
        Infers whether a data point is in the training set by using the LIRA membership inference attack.

        :param dataset: The target data points to be inferred.
        :return: The inferred membership status of the data point.
        """
        LIRAUtil.log(self.aux_info, "Start membership inference...", print_flag=True)

        set_seed(self.aux_info.seed)

        shadow_model = self.target_model_access.get_untrained_model()
        # concatenate the target dataset and the auxiliary dataset
        shadow_target_concat_set = ConcatDataset([self.auxiliary_dataset, dataset])
        LIRAUtil.train_shadow_models(shadow_model, shadow_target_concat_set, info=self.aux_info)

        # given the model, calculate the score and generate the kept index data
        self.shadow_scores, self.shadow_keeps = LIRAUtil.process_shadow_models(self.aux_info,
                                                                                shadow_target_concat_set,
                                                                                shadow_model)
        # Convert the list of tensors to a single tensor
        self.shadow_scores = torch.cat(self.shadow_scores, dim=0)
        self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)


        # obtaining target_score, which is the score of target datapoints on the target model
        target_scores = LIRAUtil.process_target_model(self.target_model_access, self.aux_info,
                                                    shadow_target_concat_set)
        target_scores = torch.cat(target_scores, dim=0)

        if self.aux_info.online: # online
            predictions = LIRAUtil.lira_mia(np.array(self.shadow_keeps), np.array(self.shadow_scores),
                                            np.array(target_scores), fix_variance=self.aux_info.fix_variance)

            predictions = -predictions[-len(dataset):]

        else: # offline
            predictions = LIRAUtil.lira_mia_offline(np.array(self.shadow_keeps), np.array(self.shadow_scores),
                                                    np.array(target_scores), fix_variance=self.aux_info.fix_variance)

            predictions = predictions[-len(dataset):]

        LIRAUtil.log(self.aux_info, "Finish membership inference...", print_flag=True)
        
        # return the predictions on the target data
        return predictions
