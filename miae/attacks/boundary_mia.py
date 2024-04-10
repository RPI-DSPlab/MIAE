# This code implements "Label-Only Membership Inference Attacks", PMLR 2021
# This code is based on the implementation on https://github.com/cchoquette/membership-inference
# Note that this file only implements their Boundary + Translation attack, which is their best performing one

import copy
import logging
import os
from typing import List

import numpy as np
import torch
from scipy.ndimage import interpolation
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from art.attacks.evasion import HopSkipJump
from sklearn.metrics import roc_curve
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import dataset_split

class AttackModel(nn.Module):
    def __init__(self, aug_type='n'):
        super(AttackModel, self).__init__()
        if aug_type == 'n':
            self.x1 = nn.Linear(in_features=64, out_features=64)
            self.x_out = nn.Linear(in_features=64, out_features=2)
        elif aug_type == 'r' or aug_type == 'd':
            self.x1 = nn.Linear(in_features=10, out_features=10)
            self.x2 = nn.Linear(in_features=10, out_features=10)
            self.x_out = nn.Linear(in_features=10, out_features=2)
        else:
            raise ValueError(f"aug_type={aug_type} is not valid.")
        self.x_activation = nn.Softmax(dim=1)

    def forward(self, x):
        if hasattr(self, 'x2'):
            x = F.relu(self.x1(x))
            x = F.relu(self.x2(x))
        else:
            x = F.relu(self.x1(x))
        x = self.x_out(x)
        x = self.x_activation(x)
        return x


class BoundaryAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of the auxiliary information for the Boundary attack (Label-only attack).
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
        self.input_dim = config.get("input_dim", [None, 32, 32, 3])
        self.n_classes = config.get("n_classes", 10)


        # -- attack model parameters --
        self.attack_model = attack_model
        self.num_attack_epochs = config.get("num_attack_epochs", self.epochs)
        self.attack_batch_size = config.get("attack_batch_size", self.batch_size)
        self.attack_lr = config.get("attack_lr", 0.01)
        self.attack_train_ratio = config.get("attack_train_ratio", 0.9)
        self.attack_epochs = config.get("attack_epochs", self.epochs)

        # -- other parameters --
        self.save_path = config.get("save_path", "boundary")
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.shadow_model_path = config.get("shadow_model_path", f"{self.save_path}/shadow_models")
        self.attack_dataset_path = config.get("attack_dataset_path", f"{self.save_path}/attack_dataset")
        self.attack_model_path = config.get("attack_model_path", f"{self.save_path}/attack_models")
        self.cos_scheduler = config.get("cos_scheduler", True)  # use cosine annealing scheduler for shadow model

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)
        if self.log_path is not None:
            self.boundary_logger = logging.getLogger('boundary_logger')
            self.boundary_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/boundary.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.boundary_logger.addHandler(fh)


class BoundaryModelAccess(ModelAccess):
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


class BoundaryUtil(MIAUtils):

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
    def dists(cls, model, ds, aux_info: BoundaryAuxiliaryInfo, attack="HSJ"):
        device = aux_info.device
        input_dim = aux_info.input_dim
        n_classes = aux_info.n_classes

        model.eval()

        acc = []
        dist_adv = []

        if attack == "CW":
            raise NotImplementedError("CW attack not implemented yet")
        elif attack == "HSJ":
            def generate_adversarial_example(model, x, y):
                attack = HopSkipJump(model=model, targeted=False, norm=np.inf, max_iter=1000, max_eval=10000,
                                     init_eval=100, init_size=100, verbose=False)
                x_adv = attack.generate(x, y)
                return x_adv

        else:
            raise ValueError("Unknown attack {}".format(attack))

        data_loader = DataLoader(ds, batch_size=1, shuffle=False)

        num_samples = 0
        for batch_idx, (xbatch, ybatch) in enumerate(data_loader):
            xbatch, ybatch = xbatch.to(device), ybatch.to(device)
            ybatch_onehot = torch.eye(n_classes)[ybatch].to(device)

            with torch.no_grad():
                output = model(xbatch)
                y_pred = torch.argmax(output, dim=1)
                correct = (y_pred == ybatch).cpu().numpy()
                acc.extend(correct)

                if correct:
                    # Generate adversarial examples
                    x_adv = generate_adversarial_example(model, xbatch, ybatch_onehot)
                else:
                    x_adv = xbatch

                # Compute distances
                d = torch.norm(x_adv - xbatch, p=2, dim=(1, 2, 3)).cpu().numpy()
                dist_adv.extend(d)

            num_samples += xbatch.size(0)
            print("Processed {} examples".format(num_samples))
            if num_samples >= len(ds):
                break

        return dist_adv

    @classmethod
    def distance_augmentation_process(cls, model, data, aux_info: BoundaryAuxiliaryInfo,
                                     attack_type='d', augment_kwarg=1) -> np.array:
        """process data for distance augmentation attack's training and inference.

        Args:
          model: model to approximate distances on (attack).
          train_set: the training set for the shadow model
          test_set: the test set for the shadow model
          attack_type: either 'd' or 'r' for translation and rotation attacks, respectively.
          augment_kwarg: the kwarg for each augmentation. If rotations, augment_kwarg defines the max rotation, with n=2r+1
          rotated images being used. If translations, then 4n+1 translations will be used at a max displacement of
          augment_kwarg

        Returns: data after processing for distance augmentation attack, with shape (len(data), len(augments)).

        """
        if attack_type == 'r':
            raise NotImplementedError("Rotation attack not implemented yet")
        elif attack_type == 'd':
            augments = cls.create_translates(augment_kwarg)
        else:
            raise ValueError(f"attack type_: {attack_type} is not valid.")

        processed_ds = np.zeros((len(data), len(augments)))

        for i in range(len(augments)):
            data_aug = cls.apply_augment(data, augments[i], attack_type)
            ds = TensorDataset(torch.tensor(data_aug[0]), torch.tensor(data_aug[1]))
            processed_ds[:, i] = cls.dists(model, ds, aux_info, attack="HSJ")
        return processed_ds


class BoundaryAttack(MiAttack):
    """
    Implementation of the Boundary attack (Label-only attack) with translation.
    """

    def __init__(self, target_model_access: BoundaryModelAccess, auxiliary_info: BoundaryAuxiliaryInfo):
        """
        Initialize the Shokri attack with model access and auxiliary information.
        :param target_model_access: the model access to the target model.
        :param auxiliary_info: the auxiliary information for the Shokri attack.
        :param target_data: the target data for the Shokri attack.
        """
        super().__init__(target_model_access, auxiliary_info)
        self.aux_info = auxiliary_info
        self.target_model_access = target_model_access
        self.attack_model = None
        self.prepared = False

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
        set_seed(self.aux_info.seed)
        # 1. Train a shadow model
        train_set_len = int(len(auxiliary_dataset) * self.aux_info.shadow_train_ratio)
        test_set_len = len(auxiliary_dataset) - train_set_len
        train_set, test_set = dataset_split(auxiliary_dataset, [train_set_len, test_set_len])

        shadow_model = self.target_model_access.get_untrained_model()
        if os.path.exists(self.aux_info.shadow_model_path + '/shadow_model.pth'):
            shadow_model = torch.load(self.aux_info.shadow_model_path + '/shadow_model.pth')
        else:
            trainloader = DataLoader(train_set, batch_size=self.aux_info.shadow_batch_size, shuffle=True, num_workers=2)
            testloader = DataLoader(test_set, batch_size=self.aux_info.shadow_batch_size, shuffle=False, num_workers=2)
            shadow_model = BoundaryUtil.train_shadow_model(shadow_model, trainloader, testloader, self.aux_info)
            torch.save(shadow_model, self.aux_info.shadow_model_path + '/shadow_model.pth')

        # 2. Generate distance from the shadow model and auxiliary dataset
        dist_in = BoundaryUtil.distance_augmentation_process(shadow_model, train_set, self.aux_info, attack_type='d', augment_kwarg=1)
        dist_out = BoundaryUtil.distance_augmentation_process(shadow_model, test_set, self.aux_info, attack_type='d', augment_kwarg=1)

        # 3. Train an attack model
        attack_model = self.aux_info.attack_model()
        train_set = TensorDataset(torch.tensor(dist_in), torch.tensor([1] * len(dist_in)))
        test_set = TensorDataset(torch.tensor(dist_out), torch.tensor([0] * len(dist_out)))
        attack_dataset = torch.utils.data.ConcatDataset([train_set, test_set])
        attack_train_loader = DataLoader(attack_dataset, batch_size=self.aux_info.attack_batch_size, shuffle=True)
        self.attack_model = BoundaryUtil.train_attack_model(attack_model, attack_train_loader, self.aux_info)

        self.prepared = True

    def infer(self, data) -> np.ndarray:
        """
        Infer the membership of the target data.
        1. process the data for distance augmentation attack
        2. infer the membership of the target data with the attack model
        """
        super().infer(data)
        if not self.prepared:
            raise ValueError("The attack has not been prepared yet!")

        # 1. process the data for distance augmentation attack
        dist_target = BoundaryUtil.distance_augmentation_process(self.target_model_access.model, data, self.aux_info, attack_type='d', augment_kwarg=1)
        self.target_model_access.to_device(self.aux_info.device)
        target_pred = self.attack_model(torch.tensor(dist_target).to(self.aux_info.device)).detach().cpu().numpy()
        return target_pred[1]




