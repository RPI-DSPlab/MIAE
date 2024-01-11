# This code implements "Membership Inference Attacks against Machine Learning Models" by Shokri et al.
# https://arxiv.org/abs/1610.05820
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_curve
from tqdm import tqdm

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from mia.utils import datasets
from mia.utils import models
from mia.utils.set_seed import set_seed


class ShokriAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of the auxiliary information for the Shokri attack.
    """

    def __init__(self, config):
        """
        Initialize the auxiliary information with default config.
        :param config: a dictionary containing the configuration for auxiliary information.
        """
        super().__init__(config)
        # ---- initialize auxiliary information with default values ----
        self.seed = config.get("seed", 0)
        # -- Shadow model parameters --
        self.num_shadow_models = config.get("num_shadow_models", 10)
        self.num_shadow_epochs = config.get("num_shadow_epochs", 160)
        self.shadow_batch_size = config.get("shadow_batch_size", 500)
        self.shadow_lr = config.get("shadow_lr", 0.001)
        self.shadow_train_ratio = config.get("shadow_train_ratio", 0.5)  # 0.5 for a balanced prior for membership

        # -- attack model parameters --
        self.num_attack_epochs = config.get("num_attack_epochs", 160)
        self.attack_batch_size = config.get("attack_batch_size", 128)
        self.attack_lr = config.get("attack_lr", 0.001)
        self.attack_train_ratio = config.get("attack_train_ratio", 0.9)

        # -- other parameters --
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.shadow_model_path = config.get("shadow_model_path", "shokri/shadow_models")
        self.attack_dataset_path = config.get("attack_dataset_path", "shokri/attack_dataset")
        self.attack_model_path = config.get("attack_model_path", "shokri/attack_model")


class ShokriModelAccess(ModelAccess):
    """
    Implementation of model access for Shokri attack.
    """

    def __init__(self, model, access_type: ModelAccessType):
        """
        Initialize model access with model and access type.
        :param model: the target model.
        :param access_type: the type of access to the target model.
        """
        super().__init__(model, access_type)

    def to_device(self, device):
        self.model.to(device)

    def get_model_architecture(self):
        """
        Get the un-initialized model architecture of the target model.
        :return: the un-initialized model architecture.
        """
        return copy.deepcopy(self.model).reset_parameters()


class ShokriUtil:
    @classmethod
    def train_shadow_model(cls, shadow_model, shadow_train_loader, shadow_test_loader, shadow_epochs, shadow_lr,
                           device):
        """
        Train the shadow model.
        :param shadow_model: the shadow model.
        :param shadow_train_loader: the shadow training data loader.
        :param shadow_test_loader: the shadow test data loader.
        :param shadow_epochs: the number of epochs for training the shadow model.
        :param shadow_lr: the learning rate for training the shadow model.
        :param device: the device for training the shadow model.
        :return: the trained shadow model.
        """
        shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=shadow_lr)
        shadow_criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(shadow_epochs)):
            shadow_model.train()
            for data, target in shadow_train_loader:
                data, target = data.to(device), target.to(device)
                shadow_optimizer.zero_grad()
                output = shadow_model(data)
                loss = shadow_criterion(output, target)
                loss.backward()
                shadow_optimizer.step()

            shadow_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in shadow_test_loader:
                    data, target = data.to(device), target.to(device)
                    output = shadow_model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            if epoch % 20 == 0 or epoch == shadow_epochs - 1:
                print(f"Epoch: {epoch}, Accuracy: {correct / total}, Loss: {loss.item()}")

        return shadow_model

    @classmethod
    def train_attack_model(cls, attack_model, attack_train_loader, attack_test_loader, attack_epochs, attack_lr,
                           device):
        """
        Train the attack model.
        :param attack_model: the attack model.
        :param attack_train_loader: the attack training data loader.
        :param attack_test_loader: the attack test data loader.
        :param attack_epochs: the number of epochs for training the attack model.
        :param attack_lr: the learning rate for training the attack model.
        :param device: the device for training the attack model.
        :return: the trained attack model.
        """
        attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=attack_lr)
        attack_criterion = torch.nn.BCELoss()

        for epoch in tqdm(range(attack_epochs)):
            attack_model.train()
            for data, target in attack_train_loader:
                data, target = data.to(device), target.to(device)
                attack_optimizer.zero_grad()
                output = attack_model(data)
                loss = attack_criterion(output, target)
                loss.backward()
                attack_optimizer.step()

            attack_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in attack_test_loader:
                    data, target = data.to(device), target.to(device)
                    output = attack_model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            if epoch % 20 == 0 or epoch == attack_epochs - 1:
                print(f"Epoch: {epoch}, Accuracy: {correct / total}, Loss: {loss.item()}")

        return attack_model


class AttackTrainingSet(Dataset):
    def __init__(self, predictions, class_labels, in_out):
        self.predictions = predictions  # Prediction values
        self.class_labels = class_labels  # Class labels
        self.in_out = in_out  # "in" or "out" indicator

        # Ensure all inputs have the same length
        assert len(predictions) == len(class_labels) == len(in_out), "Lengths of inputs should match"

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        prediction = self.predictions[idx]
        class_label = self.class_labels[idx]
        in_out_indicator = self.in_out[idx]

        return {'prediction': prediction, 'class_label': class_label, 'in_out': in_out_indicator}


class ShokriAttack(MiAttack):
    """
    Implementation of the Shokri attack.
    """

    def __init__(self, target_model_access: ShokriModelAccess, auxiliary_info: ShokriAuxiliaryInfo, target_data=None):
        """
        Initialize the Shokri attack with model access and auxiliary information.
        :param target_model_access: the model access to the target model.
        :param auxiliary_info: the auxiliary information for the Shokri attack.
        :param target_data: the target data for the Shokri attack.
        """
        super().__init__(target_model_access, auxiliary_info, target_data)
        self.attack_dataset = None
        self.shadow_models = []
        self.attack_model = None
        self.shadow_train_loader = None
        self.shadow_test_loader = None
        self.attack_train_loader = None
        self.attack_test_loader = None
        self.auxiliary_info = auxiliary_info
        self.target_model_access = target_model_access

        # directories:
        for dir in [self.auxiliary_info.shadow_model_path, self.auxiliary_info.attack_model_path]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.prepared = False  # this flag indicates whether the attack has been prepared

    def prepare(self, auxiliary_dataset, attack_model_architecture=None):
        """
        Prepare the attack.
        :param auxiliary_dataset: the auxiliary dataset (will be split into training sets )
        :param attack_model_architecture: the attack model architecture.
        """
        super().prepare(auxiliary_dataset)
        if self.prepared:
            print("The attack has already prepared!")
            return

        if attack_model_architecture is None:
            raise ValueError("The attack model architecture is not specified!")

        self.attack_model = attack_model_architecture

        # set seed
        set_seed(self.auxiliary_info.seed)

        # create shadow datasets
        sub_shadow_dataset_len = int(len(auxiliary_dataset) / self.auxiliary_info.num_shadow_models)
        sub_shadow_dataset_list = []
        for i in range(self.auxiliary_info.num_shadow_models):
            sub_shadow_dataset_list.append(
                TensorDataset(auxiliary_dataset[i * sub_shadow_dataset_len: (i + 1) * sub_shadow_dataset_len][0],
                              auxiliary_dataset[i * sub_shadow_dataset_len: (i + 1) * sub_shadow_dataset_len][1]))

        # step 1: train shadow models
        if not os.path.exists(self.auxiliary_info.attack_dataset_path):
            # if attack dataset exists, then there's no need to retrain shadow models
            in_prediction_set_pred = []
            in_prediction_set_label = []
            out_prediction_set_pred = []
            out_prediction_set_label = []
            for i in range(self.auxiliary_info.num_shadow_models):
                # train k shadows models to build attack dataset
                print(f"Training shadow model {i}")
                model_name = f"shadow_model_{i}.pt"
                model_path = os.path.join(self.auxiliary_info.shadow_model_path, model_name)

                shadow_model_i = self.target_model_access.get_model_architecture()
                shadow_model_i.to(self.auxiliary_info.device)
                train_len = int(len(sub_shadow_dataset_list[i]) * self.auxiliary_info.shadow_train_ratio)
                test_len = len(sub_shadow_dataset_list[i]) - train_len
                shadow_train_dataset, shadow_test_dataset = torch.utils.data.random_split(sub_shadow_dataset_list[i],
                                                                                          [train_len, test_len])
                shadow_train_dataset = shadow_train_dataset.dataset
                shadow_test_dataset = shadow_test_dataset.dataset

                shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=self.auxiliary_info.shadow_batch_size,
                                                 shuffle=True)
                shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=self.auxiliary_info.shadow_batch_size,
                                                shuffle=True)
                shadow_model_i = ShokriUtil.train_shadow_model(shadow_model_i, shadow_train_loader, shadow_test_loader,
                                                               self.auxiliary_info.num_shadow_epochs,
                                                               self.auxiliary_info.shadow_lr,
                                                               self.auxiliary_info.device)
                torch.save(shadow_model_i.state_dict(), model_path)

                # building the attack dataset
                for data, target in shadow_train_loader:
                    data, target = data.to(self.auxiliary_info.device), target.to(self.auxiliary_info.device)
                    output = shadow_model_i(data)
                    in_prediction_set_pred.append(output.cpu().detach().numpy().flatten())
                    in_prediction_set_label.append(target.cpu().detach().numpy().flatten())

                    print(f"shape of output after flatten: {output.cpu().detach().numpy().flatten().shape}")
                    print(f"shape of target after flatten: {target.cpu().detach().numpy().flatten().shape}")

                for data, target in shadow_test_loader:
                    data, target = data.to(self.auxiliary_info.device), target.to(self.auxiliary_info.device)
                    output = shadow_model_i(data)
                    out_prediction_set_pred.append(output.cpu().detach().numpy().flatten())
                    out_prediction_set_label.append(target.cpu().detach().numpy().flatten())


            # step 2: create attack dataset for attack model training
            in_prediction_set_pred = np.array(in_prediction_set_pred).flatten()
            out_prediction_set_pred = np.array(out_prediction_set_pred).flatten()
            in_prediction_set_label = np.array(in_prediction_set_label).flatten()
            out_prediction_set_label = np.array(out_prediction_set_label).flatten()
            in_prediction_set_membership = np.ones_like(in_prediction_set_pred)
            out_prediction_set_membership = np.zeros_like(out_prediction_set_pred)

            # combine in and out prediction sets
            prediction_set_pred = np.concatenate((in_prediction_set_pred, out_prediction_set_pred))
            prediction_set_label = np.concatenate((in_prediction_set_label, out_prediction_set_label))
            prediction_set_membership = np.concatenate((in_prediction_set_membership, out_prediction_set_membership))

            # shuffle the prediction set
            shuffle_idx = np.arange(len(prediction_set_pred))
            np.random.shuffle(shuffle_idx)
            prediction_set_pred = prediction_set_pred[shuffle_idx]
            prediction_set_label = prediction_set_label[shuffle_idx]
            prediction_set_membership = prediction_set_membership[shuffle_idx]

            # build the dataset for attack model training
            self.attack_dataset = AttackTrainingSet(prediction_set_pred, prediction_set_label,
                                                    prediction_set_membership)
            torch.save(self.attack_dataset, self.auxiliary_info.attack_dataset_path)

        # step 3: train attack model
        self.attack_dataset = torch.load(self.auxiliary_info.attack_dataset_path)
        train_len = int(len(self.attack_dataset) * self.auxiliary_info.attack_train_ratio)
        test_len = len(self.attack_dataset) - train_len

        attack_train_dataset, attack_test_dataset = torch.utils.data.random_split(self.attack_dataset,
                                                                                  [train_len, test_len])
        attack_train_dataset = attack_train_dataset.dataset
        attack_test_dataset = attack_test_dataset.dataset

        self.attack_train_loader = DataLoader(attack_train_dataset, batch_size=self.auxiliary_info.attack_batch_size,
                                              shuffle=True)
        self.attack_test_loader = DataLoader(attack_test_dataset, batch_size=self.auxiliary_info.attack_batch_size,
                                                shuffle=True)

        # train the attack model
        self.attack_model.to(self.auxiliary_info.device)
        self.attack_model = ShokriUtil.train_attack_model(self.attack_model, self.attack_train_loader,
                                                           self.attack_test_loader,
                                                           self.auxiliary_info.num_attack_epochs,
                                                           self.auxiliary_info.attack_lr,
                                                           self.auxiliary_info.device)
        torch.save(self.attack_model.state_dict(), self.auxiliary_info.attack_model_path)

        self.prepared = True

    def infer(self, target_data):
        """
        Infer the membership of the target data.
        """
        super().infer(target_data)
        if not self.prepared:
            raise ValueError("The attack has not been prepared!")
        self.attack_model.to(self.auxiliary_info.device)
        self.attack_model.eval()
        with torch.no_grad():
            output = self.attack_model(target_data)
            output = output.cpu().detach().numpy().flatten()
        return output

