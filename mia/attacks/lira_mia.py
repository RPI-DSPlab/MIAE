# This code implements "Membership Inference Attacks From First Principles", S&P 2022
# The code is based on the code from
# https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021
import logging
import os

import numpy as np
import scipy
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import transforms

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from utils.datasets.loader import load_dataset
from utils.models.loader import init_network
from utils.optim import trainer
from utils.optim.trainer import load_model
from utils.set_seed import set_seed


# from attack_classifier import AttackClassifier # no classifer needed for LIRA

class LiraModelAccess(ModelAccess):
    """
    Your implementation of ModelAccess for Lira.
    """

    def __init__(self, model, access_type: ModelAccessType):
        """
        Initialize LiraModelAccess.
        """
        super().__init__(model, access_type)
        self.model = model
        self.model.eval()

    def get_signal(self, data):
        """
        Generates logits for a dataset given a model.

        Args:
        model (torch.nn.Module): The PyTorch model to generate logits.
        data: a data point
        """
        logits = []

        with torch.no_grad():
            images, _ = data
            images = images

            batch_logits = []
            for aug in [images, images.flip(2)]:
                pad = torch.nn.ReflectionPad2d(2)
                aug_pad = pad(aug)
                this_x = aug_pad[:, :, :32, :32]

                outputs = self.model(this_x)
                batch_logits.append(outputs)

            logits.append(torch.stack(batch_logits).permute(1, 0, 2))

        return torch.cat(logits).unsqueeze(1)  # should be [num_samples, 2, num_classes]


class LiraAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of AuxiliaryInfo for Lira.
    """

    def __init__(self, config):
        """
        Initialize LiraAuxiliaryInfo with a configuration dictionary.
        """
        self.config = config

        # Model architecture parameters
        self.arch = config.get('arch', "wrn28-2")
        self.init_weights = config.get('init_weights', True)

        # Training parameters
        self.weight_decay = config.get('weight_decay', 0.01)
        self.decay = config.get('decay', 0.9999)
        self.target_seed_base = config.get('target_seed_base', 24)  # the seed begin number for target model
        self.shadow_seed_base = config.get('shadow_seed_base', 100)  # the seed begin number for shadow model
        self.epochs = config.get('epochs', 140)
        self.device = config.get('device', "cpu")

        # Model saving and loading parameters
        self.path = config.get('path', None)

        # Auxiliary info for LIRA
        self.dataset_name = config.get('dataset_name', None)
        self.dataset_dir = config.get('dataset_dir', f"./data/{self.dataset_name}")
        self.num_shadow_models = config.get('num_shadow_models', None)
        self.num_target_models = config.get('num_target_models', 1)
        self.shadow_path = config.get('shadow_path', f"./weights/shadow/{self.arch}/")
        self.target_path = config.get('target_path', f"./weights/target/{self.arch}/")


def _split_data(fullset, expid, iteration_range, is_shadow):
    if is_shadow:  # case: for shadow model
        keep = np.random.uniform(0, 1, size=(iteration_range, len(fullset)))
        order = keep.argsort(0)
        keep = order < int(.5 * iteration_range)
        keep = np.array(keep[expid], dtype=bool)
        return np.where(keep)[0], np.where(~keep)[0]
    else:  # case: target model: random split
        keep = np.random.uniform(0, 1, size=len(fullset)) <= 0.5
        return np.where(keep)[0], np.where(~keep)[0]


class LIRAUtil:
    @classmethod
    def _prepare_data(cls, info: LiraAuxiliaryInfo):
        """
        Prepares the data by loading the dataset and applying transformations.

        Args:
        info (LiraAuxiliaryInfo): The auxiliary info instance containing all the necessary information.
        """
        # Constants
        DATA_TRANSFORM = transforms.Compose([
            transforms.RandomCrop(32, padding=(4, 4), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Prepare the data
        dataset = load_dataset(dataset_name=info.dataset_name, data_path=info.dataset_dir,
                               train_transform=DATA_TRANSFORM, test_transform=DATA_TRANSFORM, target_transform=None)

        # Combine trainset and testset
        return dataset

    @classmethod
    def _make_directory_if_not_exists(cls, dir_path):
        """
        Checks if a directory exists and, if not, creates it.

        Args:
        dir_path (str): The path of the directory to be checked/created.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @classmethod
    def train_models(cls, info: LiraAuxiliaryInfo, is_shadow=True):
        """
        Trains the models (either shadow or target models).

        Args:
        info (LiraAuxiliaryInfo): The auxiliary info instance containing all the necessary information.
        is_shadow (bool): If true, shadow models are trained. Otherwise, target models are trained.
        """
        # init
        mode = "shadow" if is_shadow else "target"
        iteration_range = info.num_shadow_models if is_shadow else info.num_target_models
        seed_base = info.shadow_seed_base if is_shadow else info.target_seed_base
        set_seed(seed_base)
        device = torch.device(info.device)
        dataset = cls._prepare_data(info)
        fullset = ConcatDataset([dataset.train_set, dataset.test_set])

        for expid in range(iteration_range):
            # Define the directory path
            folder_name = expid if is_shadow else expid + seed_base
            dir_path = f"./weights/{mode}/{info.arch}/{folder_name}"
            log_path = f"./logs/{mode}/{info.arch}"

            # Check if the directory exists and create
            cls._make_directory_if_not_exists(dir_path)
            cls._make_directory_if_not_exists(log_path)

            set_seed(expid + seed_base)

            # split the data
            shadow_train_indices, shadow_out_indices = _split_data(fullset, expid, iteration_range, is_shadow)

            # Create the data loaders for training and testing
            shadow_train_loader = DataLoader(Subset(fullset, shadow_train_indices), batch_size=256, shuffle=True)
            shadow_out_loader = DataLoader(Subset(fullset, shadow_out_indices), batch_size=256, shuffle=False)

            # Prepare to train the target model
            target_model = init_network(arch=info.arch, init_weights=True).to(device)
            optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001, weight_decay=0.0005)
            scheduler = CosineAnnealingLR(optimizer, T_max=info.epochs, eta_min=0)

            # Logging
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[
                                    logging.FileHandler(f"{log_path}/log_expid_{expid}.log"),
                                    logging.StreamHandler()
                                ])

            for epoch in range(1, info.epochs):
                loss, _ = trainer.train(target_model, device, shadow_train_loader, optimizer, scheduler)
                acc = trainer.test(target_model, device, shadow_out_loader)
                logging.info(f"Train Target Model: {epoch}/{info.epochs}: TRAIN loss: {loss:.8f}; TEST acc: {acc:.6f}")

            # save model
            trainer.save_model(target_model, f"{dir_path}/{mode}.pth")
            # save keep
            is_in_train = np.full(len(fullset), False)
            is_in_train[shadow_train_indices] = True
            np.save(f"{dir_path}/keep.npy", is_in_train)

    @classmethod
    def lira_mia(cls, keep, scores, check_scores, in_size=100000, out_size=100000,
                 fix_variance=False):
        """
        Implements the core logic of the LIRA membership inference attack.

        Args:
        keep (np.ndarray): An array indicating which samples to keep.
        scores (np.ndarray): An array containing the scores of the samples.
        check_scores (np.ndarray): An array containing the scores of the samples for target model.
        in_size (int): The number of samples to keep from the input.
        out_size (int): The number of samples to keep from the output.
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

        if fix_variance:
            std_in = np.std(dat_in)
            std_out = np.std(dat_in)
        else:
            std_in = np.std(dat_in, 1)
            std_out = np.std(dat_out, 1)

        prediction = []

        for sc in check_scores:
            pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out

            prediction.extend(score.mean(1))

        return prediction

    @classmethod
    def _generate_logits(cls, model, data_loader, device):
        """
        Generates logits for a dataset given a model.

        Args:
        model (torch.nn.Module): The PyTorch model to generate logits.
        data_loader (torch.utils.data.DataLoader): The DataLoader for the dataset.
        device (str): The device (cpu or cuda) where the computations will take place.
        """
        model.eval()
        logits = []

        with torch.no_grad():
            for batch in data_loader:
                images, _ = batch
                images = images.to(device)

                batch_logits = []
                for aug in [images, images.flip(2)]:
                    pad = torch.nn.ReflectionPad2d(2)
                    aug_pad = pad(aug)
                    this_x = aug_pad[:, :, :32, :32]

                    outputs = model(this_x)
                    batch_logits.append(outputs)

                logits.append(torch.stack(batch_logits).permute(1, 0, 2))

        return torch.cat(logits).unsqueeze(1)  # should be [num_samples, 2, num_classes]

    @classmethod
    def process_models(cls, info: LiraAuxiliaryInfo, target_model_access: ModelAccess,
                       is_shadow=True, threshold_acc=0.5):
        """
        Loads the models and calculates the scores for each model.

        Args:
        info (LiraAuxiliaryInfo): The auxiliary info instance containing all the necessary information.
        is_shadow (bool): If true, shadow models are processed. Otherwise, target models are processed.
        threshold_acc (float): The accuracy threshold for skipping records.
        """
        dataset = cls._prepare_data(info)
        fullset = ConcatDataset([dataset.train_set, dataset.test_set])
        fullsetloader = torch.utils.data.DataLoader(fullset, batch_size=20, shuffle=False, num_workers=8)

        # Combine the targets from trainset and testset
        fullset_targets = dataset.train_set.targets + dataset.test_set.targets

        score_list = []
        keep_list = []
        model_path = info.shadow_path if is_shadow else info.target_path
        model_locations = os.listdir(model_path)

        for index, dir_name in enumerate(model_locations, start=1):
            seed_folder = os.path.join(model_path, dir_name)
            if os.path.isdir(seed_folder):
                model_path = os.path.join(seed_folder, f"{'shadow' if is_shadow else 'target'}.pth")
                if os.path.isfile(model_path):
                    print(f"load model [{index}/{len(model_locations)}]: {model_path}")
                    model = load_model(info.arch, path=model_path).to(info.device)
                    scores, mean_acc = cls._calculate_score(cls._generate_logits(model,
                                                                                 fullsetloader,
                                                                                 info.device).cpu().numpy(),
                                                            fullset_targets)
                    if mean_acc < threshold_acc: continue  # model is too bad, skip this record

                    # Convert the numpy array to a PyTorch tensor and add a new dimension
                    scores = torch.unsqueeze(torch.from_numpy(scores), 0)
                    score_list.append(scores)

                keep_path = os.path.join(seed_folder, "keep.npy")
                if os.path.isfile(keep_path):
                    keep = torch.unsqueeze(torch.from_numpy(np.load(keep_path)), 0)
                    keep_list.append(keep)

        return score_list, keep_list

    @classmethod
    def _calculate_score(cls, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the score for each prediction.

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
        print(f'mean accuracy: {mean_acc:.4f}')

        # Zero out the true class predictions
        predictions[np.arange(COUNT), :, :, labels[:COUNT]] = 0

        # Calculate sum of predictions for incorrect classes
        y_wrong = np.sum(predictions, axis=3)

        # Calculate log-odds of correct versus incorrect predictions
        score = (np.log(y_true.mean(axis=1) + 1e-45) - np.log(y_wrong.mean(axis=1) + 1e-45))

        # CIFAR 10 --> (50000, 0, 1, 10)
        # SCORE --> (50000, 1, 10)
        return score, mean_acc


class LiraMiAttack(MiAttack):
    """
    Implementation of MiAttack for Lira.
    """

    def __init__(self, target_model_access: LiraModelAccess, auxiliary_info: LiraAuxiliaryInfo, target_data=None):
        """
        Initialize LiraMiAttack.
        """
        super().__init__(target_model_access, auxiliary_info, target_data)
        self.shadow_scores = self.shadow_keeps = self.target_scores = self.target_keeps = None
        self.auxiliary_info = auxiliary_info
        self.config = self.auxiliary_info.config

    def prepare(self, attack_config: dict):
        """
        Prepares for the attack by training models and generating the score and kept index data.

        Args:
        attack_config (dict): The attack configuration dictionary.
        """
        LIRAUtil.train_models(info=self.auxiliary_info, is_shadow=True)

        # given the model, calculate the score and generate the kept index data
        self.shadow_scores, self.shadow_keeps = LIRAUtil.process_models(self.auxiliary_info.shadow_path,
                                                                        self.target_model_access,
                                                                        is_shadow=True)

        # Convert the list of tensors to a single tensor
        self.shadow_scores = torch.cat(self.shadow_scores, dim=0)  # (20, 60000, 2)
        self.shadow_keeps = torch.cat(self.shadow_keeps, dim=0)  # (20, 60000)

    def infer(self, target_data):
        """
        Infers whether a data point is in the training set by using the LIRA membership inference attack.

        Args:
        target_data (torch.utils.data.Dataset): The target dataset.
        """
        if self.shadow_scores is None or \
                self.shadow_keeps is None or \
                self.target_scores is None or \
                self.target_keeps is None:
            raise "You should init the model by calling prepare() first"

        return LIRAUtil.lira_mia(np.array(self.shadow_scores), np.array(self.shadow_keeps), target_data)
