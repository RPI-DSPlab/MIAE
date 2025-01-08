# This code implements "Label-Only Membership Inference Attacks", PMLR 2021
# This code is based on the implementation on https://github.com/cchoquette/membership-inference
# Note that this file only implements their Augmentation (Translation) attack

import copy
import logging
import os
from typing import List

import numpy as np
import torch
from scipy.ndimage import interpolation
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils, AttackTrainingSet
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import dataset_split, get_xy_from_dataset


class AttackModel(nn.Module):
    def __init__(self, aug_type='d', augment_kwarg=2):
        super(AttackModel, self).__init__()
        input_dim = len(AugUtil.create_translates(augment_kwarg))
        if aug_type == 'n':
            self.x1 = nn.Linear(in_features=64, out_features=64)
            self.x_out = nn.Linear(in_features=64, out_features=2)
        elif aug_type == 'r' or aug_type == 'd':
            self.x1 = nn.Linear(in_features=input_dim, out_features=input_dim)
            self.x2 = nn.Linear(in_features=input_dim, out_features=input_dim)
            self.x_out = nn.Linear(in_features=input_dim, out_features=2)
        else:
            raise ValueError(f"aug_type={aug_type} is not valid.")
        self.x_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        if hasattr(self, 'x2'):
            x = F.relu(self.x1(x))
            x = F.relu(self.x2(x))
        else:
            x = F.relu(self.x1(x))
        x = self.x_out(x)
        x = self.x_activation(x)
        return x


class AugAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of the auxiliary information for the Augmentation attack (Label-only attack).
    """

    def __init__(self, config, attack_model=AttackModel):
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
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 100)
        self.momentum = config.get("momentum", 0.9)
        self.weight_decay = config.get("weight_decay", 0.0001)
        # -- Shadow model parameters --
        self.num_shadow_epochs = config.get("num_shadow_epochs", self.epochs)
        self.shadow_batch_size = config.get("shadow_batch_size", self.batch_size)
        self.shadow_lr = config.get("shadow_lr", self.batch_size)
        self.shadow_train_ratio = config.get("shadow_train_ratio", 0.5)  # 0.5 for a balanced prior for membership
        # -- attack parameters --
        self.dist_max_sample = config.get("dist_max_sample", 100)
        self.input_dim = config.get("input_dim", [3, 32, 32])
        self.n_classes = config.get("n_classes", 10)
        self.augment_kwarg = config.get("augment_kwarg", 2)

        # -- attack model parameters --
        self.attack_model = attack_model
        self.num_attack_epochs = config.get("num_attack_epochs", self.epochs)
        self.attack_batch_size = config.get("attack_batch_size", self.batch_size)
        self.attack_lr = config.get("attack_lr", 0.01)
        self.attack_train_ratio = config.get("attack_train_ratio", 1)
        self.attack_epochs = config.get("attack_epochs", self.epochs)
        # -- other parameters --
        self.save_path = config.get("save_path", "boundary")
        self.device = config.get("device", 'cuda:1' if torch.cuda.is_available() else 'cpu')
        self.shadow_model_path = config.get("shadow_model_path", f"{self.save_path}/shadow_models")
        self.attack_dataset_path = config.get("attack_dataset_path", f"{self.save_path}/attack_dataset")
        self.attack_model_path = config.get("attack_model_path", f"{self.save_path}/attack_models")
        self.cos_scheduler = config.get("cos_scheduler", True)  # use cosine annealing scheduler for shadow model

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)
        if self.log_path is not None:
            self.logger = logging.getLogger('boundary_logger')
            self.logger.setLevel(logging.INFO)
            if not os.path.exists(self.log_path):
                os.mkdir(self.log_path)
            fh = logging.FileHandler(self.log_path + '/boundary.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


class AugModelAccess(ModelAccess):
    """
    Implementation of model access for Boundary attack.
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType = ModelAccessType.LABEL_ONLY):
        """
        Initialize model access with model and access type.
        :param model: the target model.
        :param access_type: the type of access to the target model.
        """
        super().__init__(model, untrained_model, access_type)


class AugUtil(MIAUtils):

    # ----------------- helper functions for distance augmentation attacks -----------------
    @classmethod
    def apply_augment(cls, ds, augment, type_):
        """Applies an augmentation from create_rotates or create_translates.

        Args:
          ds: tuple of (images, labels) describing a dataset. Images should be 4D of (N,H,W,C) where N is total images.
          augment: the augment to apply. (one element from augments returned by create_rotates/translates)
          type_: attack type, either 'd' or 'r'

        Returns:

        """
        if type_ == 'd':
            ds = (interpolation.shift(ds[0], augment, mode='nearest'), ds[1])
        else:
            ds = (interpolation.rotate(ds[0], augment, (1, 2), reshape=False), ds[1])
        return ds

    @classmethod
    def create_translates(cls, d):
        """Creates vector of translation displacements compatible with scipy' translate.

        Args:
          d: param d for translation augmentation attack. Defines max displacement by d. Leads to 4*d+1 total images per sample.

        Returns: vector of translation displacements compatible with scipy' translate.
        """
        if d is None:
            return None

        def all_shifts(mshift):
            if mshift == 0:
                return [(0, 0, 0, 0)]
            all_pairs = []
            start = (0, mshift, 0, 0)
            end = (0, mshift, 0, 0)
            vdir = -1
            hdir = -1
            first_time = True
            while (start[1] != end[1] or start[2] != end[2]) or first_time:
                all_pairs.append(start)
                start = (0, start[1] + vdir, start[2] + hdir, 0)
                if abs(start[1]) == mshift:
                    vdir *= -1
                if abs(start[2]) == mshift:
                    hdir *= -1
                first_time = False
            all_pairs = [(0, 0, 0, 0)] + all_pairs  # add no shift
            return all_pairs

        translates = all_shifts(d)
        return translates

    @classmethod
    def check_correct(cls, dataloader, model: AugModelAccess, device) -> np.ndarray:
        """
        Run inference on the model and return the correctness of the predictions.
        :param dataloader: the dataloader for the dataset.
        :param model: the model to run inference on.
        :param device: the device to run inference on.

        Returns: the correctness of the predictions (1 if correct, 0 if incorrect).
        """
        correctness = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = model.get_signal(x)
                correct = (output == y).cpu().numpy()
                correctness.extend(correct)
        return np.array(correctness)

    @classmethod
    def augmentation_process(cls, model_access: AugModelAccess,
                             data, aux_info: AugAuxiliaryInfo, augment_kwarg=2) -> np.array:
        """process data for augmentation attack's training and inference.

        Args:
          model_access: model access to check correctness of augmentation.
          data: the dataset to process.
            aux_info: auxiliary information for the attack.
          augment_kwarg: the kwarg for each augmentation. If rotations, augment_kwarg defines the max rotation, with n=2r+1
          rotated images being used. If translations, then 4n+1 translations will be used at a max displacement of
          augment_kwarg

        Returns: data after processing for distance augmentation attack, with shape (len(data), len(augments)).

        """
        augments = cls.create_translates(augment_kwarg)
        processed_ds = np.zeros((len(data), len(augments)))
        model_access.to_device(aux_info.device)

        for i in range(len(augments)):
            AugUtil.log(aux_info, f"Processing augmentation {i + 1}/{len(augments)}", print_flag=True)
            data_x, data_y = get_xy_from_dataset(data)
            data_aug = cls.apply_augment([data_x, data_y], augments[i], 'd')
            ds = TensorDataset(torch.tensor(data_aug[0]), torch.tensor(data_aug[1]))
            loader = DataLoader(ds, batch_size=aux_info.batch_size, shuffle=False)
            correctness = cls.check_correct(loader, model_access, aux_info.device)
            processed_ds[:, i] = correctness

        return processed_ds


class AugAttack(MiAttack):
    """
    Implementation of the Boundary attack (Label-only attack) with translation.
    """

    def __init__(self, target_model_access: AugModelAccess, auxiliary_info: AugAuxiliaryInfo):
        """
        Initialize the Augmentation attack with model access and auxiliary information.
        :param target_model_access: the model access to the target model.
        :param auxiliary_info: the auxiliary information for the Shokri attack.
        :param target_data: the target data for the Shokri attack.
        """
        super().__init__(target_model_access, auxiliary_info)
        self.attack_dataset = None
        self.attack_test_loader = None
        self.attack_train_loader = None
        self.aux_info = auxiliary_info
        self.target_model_access = target_model_access
        self.attack_model = None
        self.attack_model_dict = {}
        self.prepared = False

        # directories:
        for dir in [self.aux_info.log_path, self.aux_info.save_path, self.aux_info.attack_model_path,
                    self.aux_info.shadow_model_path, self.aux_info.attack_dataset_path]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def prepare(self, auxiliary_dataset):
        """
        Prepare the attack:
        1. Train a shadow model
        2. Generate distance from the shadow model and auxiliary dataset
        3. Train an attack model

        :param auxiliary_dataset: the auxiliary dataset (will be split into training sets)
        """
        super().prepare(auxiliary_dataset)
        if self.prepared:
            print("The attack has already prepared!")
            return

        AugUtil.log(self.aux_info, "Start preparing the attack...", print_flag=True)
        set_seed(self.aux_info.seed)
        # 1. Train a shadow model
        train_set_len = int(len(auxiliary_dataset) * self.aux_info.shadow_train_ratio)
        test_set_len = len(auxiliary_dataset) - train_set_len
        train_set, test_set = dataset_split(auxiliary_dataset, [train_set_len, test_set_len])
        trainloader = DataLoader(train_set, batch_size=self.aux_info.shadow_batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(test_set, batch_size=self.aux_info.shadow_batch_size, shuffle=False, num_workers=2)

        shadow_model = self.target_model_access.get_untrained_model()
        if os.path.exists(self.aux_info.shadow_model_path + '/shadow_model.pth'):
            shadow_model = torch.load(self.aux_info.shadow_model_path + '/shadow_model.pth')
        else:
            AugUtil.log(self.aux_info, "Training shadow model", print_flag=True)

            try:
                set_seed(self.aux_info.seed)
                shadow_model.initialize_weights()
            except:
                raise NotImplementedError("the model doesn't have .initialize_weights method")

            shadow_model = AugUtil.train_shadow_model(shadow_model, trainloader, testloader, self.aux_info)
            torch.save(shadow_model, self.aux_info.shadow_model_path + '/shadow_model.pth')

        # 2. Generate different augmentation of aux dataset and their predictions on the shadow model
        if os.path.exists(self.aux_info.attack_dataset_path + '/attack_dataset.pth'):
            AugUtil.log(self.aux_info, "Loading attack dataset from file", print_flag=True)
            self.attack_dataset = torch.load(self.aux_info.attack_dataset_path + '/attack_dataset.pth')
        else:
            AugUtil.log(self.aux_info, "Generating attack dataset", print_flag=True)
            shadow_model_access = AugModelAccess(shadow_model, shadow_model)
            aug_in = AugUtil.augmentation_process(shadow_model_access, train_set, self.aux_info,
                                                  augment_kwarg=self.aux_info.augment_kwarg)
            aug_out = AugUtil.augmentation_process(shadow_model_access, test_set, self.aux_info,
                                                   augment_kwarg=self.aux_info.augment_kwarg)
            in_prediction_set_label = None
            out_prediction_set_label = None

            for _, target in trainloader:  # getting the trainset's labels
                target = target.to(self.aux_info.device)
                if in_prediction_set_label is None:  # first entry
                    in_prediction_set_label = target.cpu().detach().numpy()
                else:
                    in_prediction_set_label = np.concatenate(
                        (in_prediction_set_label, target.cpu().detach().numpy()))

            for _, target in testloader:  # getting the testset's labels
                target = target.to(self.aux_info.device)
                if out_prediction_set_label is None:  # first entry
                    out_prediction_set_label = target.cpu().detach().numpy()
                else:
                    out_prediction_set_label = np.concatenate(
                        (out_prediction_set_label, target.cpu().detach().numpy()))

            in_prediction_set_membership = np.ones(len(aug_in))
            out_prediction_set_membership = np.zeros(len(aug_out))

            # combine in and out prediction sets
            attack_set_aug = np.concatenate((aug_in, aug_out))
            attack_set_label = np.concatenate((in_prediction_set_label, out_prediction_set_label))
            attack_set_membership = np.concatenate((in_prediction_set_membership, out_prediction_set_membership))
            self.attack_dataset = AttackTrainingSet(attack_set_aug, attack_set_label, attack_set_membership)
            torch.save(self.attack_dataset, self.aux_info.attack_dataset_path + '/attack_dataset.pth')

        # 3. Train an attack model for each label
        train_len = int(len(self.attack_dataset) * self.aux_info.attack_train_ratio)
        test_len = len(self.attack_dataset) - train_len
        if self.aux_info.attack_train_ratio < 1.0:
            attack_train_dataset, attack_test_dataset = torch.utils.data.random_split(self.attack_dataset,
                                                                                      [train_len, test_len])
        else:
            attack_train_dataset = self.attack_dataset
            attack_test_dataset = None
        labels = np.unique(self.attack_dataset.class_labels)
        if len(labels) == len(os.listdir(self.aux_info.attack_model_path)):
            AugUtil.log(self.aux_info, "Loading attack models...", print_flag=True)
            for i, label in enumerate(labels):
                model = self.aux_info.attack_model()
                model.load_state_dict(torch.load(f"{self.aux_info.attack_model_path}/attack_model_{label}.pt"))
                model.to(self.aux_info.device)
                self.attack_model_dict[label] = model
        else:
            for i, label in enumerate(labels):
                AugUtil.log(self.aux_info,
                                 f"Training attack model for {i + 1}/{len(labels)} label \"{label}\" ...",
                            print_flag=True)
                # filter the dataset with the label
                attack_train_dataset_filtered = AugUtil.filter_dataset(attack_train_dataset, label)
                attack_test_dataset_filtered = AugUtil.filter_dataset(attack_test_dataset,
                                                                      label) if attack_test_dataset else None
                self.attack_train_loader = DataLoader(attack_train_dataset_filtered,
                                                      batch_size=self.aux_info.attack_batch_size,
                                                      shuffle=True)
                self.attack_test_loader = DataLoader(attack_test_dataset_filtered,
                                                     batch_size=self.aux_info.attack_batch_size,
                                                     shuffle=True) if attack_test_dataset else None
                untrained_attack_model = self.aux_info.attack_model()
                untrained_attack_model.to(self.aux_info.device)
                trained_attack_model = AugUtil.train_attack_model(untrained_attack_model,
                                                                  self.attack_train_loader,
                                                                  self.attack_test_loader,
                                                                  self.aux_info)
                self.attack_model_dict[label] = trained_attack_model
                torch.save(trained_attack_model.state_dict(),
                           f"{self.aux_info.attack_model_path}/attack_model_{label}.pt")

        self.prepared = True
        AugUtil.log(self.aux_info, "Finish preparing the attack...", print_flag=True)

    def infer(self, target_data) -> np.ndarray:
        """
        Infer the membership of the target data.
        1. process the data for augmentation attack
        2. infer the membership of the target data with the attack model
        """
        super().infer(target_data)
        set_seed(self.aux_info.seed)
        if not self.prepared:
            raise ValueError("The attack has not been prepared yet!")
        
        AugUtil.log(self.aux_info, "Start membership inference...", print_flag=True)

        # load the attack models
        labels = np.unique(self.attack_dataset.class_labels)
        for label in labels:
            if label not in self.attack_model_dict:
                model = self.aux_info.attack_model(self.aux_info.num_classes)
                model.load_state_dict(torch.load(f"{self.aux_info.attack_model_path}/attack_model_{label}.pt"))
                model.to(self.aux_info.device)
                self.attack_model_dict[label] = model

        # process the target data for augmentation attack
        if os.path.exists(self.aux_info.attack_dataset_path + '/target_data_aug.npy'):  # for testing only
            target_data_aug = np.load(self.aux_info.attack_dataset_path + '/target_data_aug.npy')
        else:
            target_data_aug = AugUtil.augmentation_process(self.target_model_access, target_data,
                                                           self.aux_info, augment_kwarg=self.aux_info.augment_kwarg)
            np.save(self.aux_info.attack_dataset_path + '/target_data_aug.npy', target_data_aug)

        # infer the membership
        self.target_model_access.to_device(self.aux_info.device)
        membership = []

        # collect the label of the target_data
        labels = [target for _, target in target_data]
        labels = np.array(labels)

        # create a dataloader for the target data
        target_dataset = TensorDataset(torch.tensor(target_data_aug), torch.tensor(labels))

        target_data_loader = DataLoader(target_dataset, batch_size=self.aux_info.attack_batch_size, shuffle=False)

        for data, target in target_data_loader:
            data = data.to(self.aux_info.device)
            for i, label in enumerate(target):
                label = label.item()
                member_pred = self.attack_model_dict[label](torch.tensor(data[i]).unsqueeze(0).to(self.aux_info.device))
                member_pred = member_pred.cpu().detach().numpy()
                membership.append(member_pred.reshape(-1))

        AugUtil.log(self.aux_info, "Finish membership inference...", print_flag=True)

        return np.array(np.transpose(membership)[1])