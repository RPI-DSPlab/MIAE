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
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils, AttackTrainingSet
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import dataset_split, get_xy_from_dataset


class AttackModel(nn.Module):
    def __init__(self, aug_type='d'):
        super(AttackModel, self).__init__()
        if aug_type == 'n':
            self.x1 = nn.Linear(in_features=64, out_features=64)
            self.x_out = nn.Linear(in_features=64, out_features=2)
        elif aug_type == 'r' or aug_type == 'd':
            self.x1 = nn.Linear(in_features=9, out_features=9)
            self.x2 = nn.Linear(in_features=9, out_features=9)
            self.x_out = nn.Linear(in_features=9, out_features=2)
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
        self.input_dim = config.get("input_dim", [3, 32, 32])
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
            fh = logging.FileHandler(self.log_path + '/boundary.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


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

        adv_attack = None
        if attack == "CW":
            raise NotImplementedError("CW attack not implemented yet")
        elif attack == "HSJ":
            adv_attack = hop_skip_jump_attack

        else:
            raise ValueError("Unknown attack {}".format(attack))

        data_loader = DataLoader(ds, batch_size=aux_info.batch_size, shuffle=False)

        num_samples = 0
        for batch_idx, (xbatch, ybatch) in enumerate(tqdm(data_loader)):
            ybatch_onehot = torch.eye(n_classes)[ybatch]
            xbatch, ybatch = xbatch.to(device), ybatch.to(device)

            with torch.no_grad():
                output = model(xbatch)
                y_pred = torch.argmax(output, dim=1)
                correct = (y_pred == ybatch).cpu().numpy()
                acc.extend(correct)
                x_adv = adv_attack(model_fn=model, x=xbatch, norm=2, verbose=False, num_iterations=5)
                for i in range(x_adv.shape[0]):
                    if correct[i]:
                        curr_adv = x_adv[i]
                    else:
                        curr_adv = xbatch[i]
                    # compute distance
                    dist_adv.append(torch.linalg.norm(curr_adv.to(device) - xbatch[i]).item())
            num_samples += xbatch.size(0)
            if num_samples >= len(ds):
                break

        return np.array(dist_adv)

    @classmethod
    def check_correct(cls, dataloader, model, device) -> np.ndarray:
        """
        Run inference on the model and return the correctness of the predictions.
        :param dataloader: the dataloader for the dataset.
        :param model: the model to run inference on.

        Returns: the correctness of the predictions (1 if correct, 0 if incorrect).
        """
        correctness = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = model(x)
                y_pred = torch.argmax(output, dim=1)
                correct = (y_pred == y).cpu().numpy()
                correctness.extend(correct)
        return np.array(correctness)

    @classmethod
    def augmentation_process(cls, model, data, aux_info: BoundaryAuxiliaryInfo,
                             attack_type='d', augment_kwarg=2) -> np.array:
        """process data for augmentation attack's training and inference.

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
        model.to(aux_info.device)

        for i in range(len(augments)):
            BoundaryUtil.log(aux_info, f"Processing augmentation {i + 1}/{len(augments)}", print_flag=True)
            data_x, data_y = get_xy_from_dataset(data)
            data_aug = cls.apply_augment([data_x, data_y], augments[i], attack_type)
            ds = TensorDataset(torch.tensor(data_aug[0]), torch.tensor(data_aug[1]))
            loader = DataLoader(ds, batch_size=aux_info.batch_size, shuffle=False)
            correctness = cls.check_correct(loader, model, aux_info.device)
            processed_ds[:, i] = correctness

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
            BoundaryUtil.log(self.aux_info, "Training shadow model", print_flag=True)
            shadow_model = BoundaryUtil.train_shadow_model(shadow_model, trainloader, testloader, self.aux_info)
            torch.save(shadow_model, self.aux_info.shadow_model_path + '/shadow_model.pth')

        # 2. Generate different augmentation of aux dataset and their predictions on the shadow model
        if os.path.exists(self.aux_info.attack_dataset_path + '/attack_dataset.pth'):
            BoundaryUtil.log(self.aux_info, "Loading attack dataset from file", print_flag=True)
            self.attack_dataset = torch.load(self.aux_info.attack_dataset_path + '/attack_dataset.pth')
        else:
            BoundaryUtil.log(self.aux_info, "Generating attack dataset", print_flag=True)
            aug_in = BoundaryUtil.augmentation_process(shadow_model, train_set, self.aux_info, attack_type='d',
                                                       augment_kwarg=2)
            aug_out = BoundaryUtil.augmentation_process(shadow_model, test_set, self.aux_info, attack_type='d',
                                                        augment_kwarg=2)
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
            BoundaryUtil.log(self.aux_info, "Loading attack models...", print_flag=True)
            for i, label in enumerate(labels):
                model = self.aux_info.attack_model()
                model.load_state_dict(torch.load(f"{self.aux_info.attack_model_path}/attack_model_{label}.pt"))
                model.to(self.aux_info.device)
                self.attack_model_dict[label] = model
        else:
            for i, label in enumerate(labels):
                BoundaryUtil.log(self.aux_info,
                                 f"Training attack model for {i + 1}/{len(labels)} label \"{label}\" ...",
                                 print_flag=True)
                # filter the dataset with the label
                attack_train_dataset_filtered = BoundaryUtil.filter_dataset(attack_train_dataset, label)
                attack_test_dataset_filtered = BoundaryUtil.filter_dataset(attack_test_dataset,
                                                                           label) if attack_test_dataset else None
                self.attack_train_loader = DataLoader(attack_train_dataset_filtered,
                                                      batch_size=self.aux_info.attack_batch_size,
                                                      shuffle=True)
                self.attack_test_loader = DataLoader(attack_test_dataset_filtered,
                                                     batch_size=self.aux_info.attack_batch_size,
                                                     shuffle=True) if attack_test_dataset else None
                untrained_attack_model = self.aux_info.attack_model()
                untrained_attack_model.to(self.aux_info.device)
                trained_attack_model = BoundaryUtil.train_attack_model(untrained_attack_model,
                                                                       self.attack_train_loader,
                                                                       self.attack_test_loader,
                                                                       self.aux_info)
                self.attack_model_dict[label] = trained_attack_model
                torch.save(trained_attack_model.state_dict(),
                           f"{self.aux_info.attack_model_path}/attack_model_{label}.pt")

        self.prepared = True

    def infer(self, target_data) -> np.ndarray:
        """
        Infer the membership of the target data.
        1. process the data for augmentation attack
        2. infer the membership of the target data with the attack model
        """
        super().infer(target_data)
        if not self.prepared:
            raise ValueError("The attack has not been prepared yet!")

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
            target_data_aug = BoundaryUtil.augmentation_process(self.target_model_access.model, target_data,
                                                                self.aux_info,
                                                                attack_type='d', augment_kwarg=2)
            np.save(self.aux_info.attack_dataset_path + '/target_data_aug.npy', target_data_aug)

        # infer the membership
        self.target_model_access.model.to(self.aux_info.device)
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

        return np.array(np.transpose(membership)[1])
