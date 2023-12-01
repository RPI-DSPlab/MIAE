# This code implements the loss trajectory based membership inference attack, published in CCS 2022 "Membership
# Inference Attacks by Exploiting Loss Trajectory".
# The code is based on the code from
# https://github.com/DennisLiu2022/Membership-Inference-Attacks-by-Exploiting-Loss-Trajectory
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch._utils import _accumulate
from tqdm import tqdm
import torch.nn.functional as F

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack


class LosstrajAuxiliaryInfo(AuxiliaryInfo):
    """
    The auxiliary information for the loss trajectory based membership inference attack.
    """

    def __init__(self, config: dict):
        """
        Initialize the auxiliary information.
        :param config: the loss trajectory.
        """
        super().__init__(config)
        self.loss_trajectory = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.get('seed', 0)
        self.batch_size = config.get('batch_size', 128)
        self.num_workers = config.get('num_workers', 2)
        self.distillation_epochs = config.get('distillation_epochs', 240)

        # directories:
        self.save_path = config.get('save_path', './losstraj_files')
        self.distill_models_path = self.save_path + '/distill_models'
        self.shadow_model_path = self.save_path + '/shadow_model.pth'

        # dataset length: it should be given as the ratio of the training dataset length w.r.t. the whole auxiliary dataset
        self.distillation_dataset_ratio = config.get('distillation_dataset_ratio', 0.5)
        self.shadow_dataset_ratio = 1 - self.distillation_dataset_ratio
        # train/test split ratio of the shadow dataset
        self.shadow_train_test_split_ratio = config.get('shadow_train_test_split_ratio', 0.7)


class LosstrajModelAccess(ModelAccess):
    """
    The model access for the loss trajectory based membership inference attack.
    """

    def __init__(self, model, model_type):
        """
        Initialize the model access.
        :param model: the target model.
        :param model_type: the type of the target model.
        """
        super().__init__(model, model_type)

    def get_model_architecture(self):
        """
        Get the un-initialized model architecture of the target model.
        :return: the un-initialized model architecture.
        """
        return copy.deepcopy(self.model).reset_parameters()


class LosstrajUtil:
    @classmethod
    def dataset_split(cls, dataset, lengths: list):
        """
        Split the dataset into subsets.
        :param dataset: the dataset.
        :param lengths: the lengths of each subset.
        """
        if sum(lengths) != len(dataset):
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        indices = list(range(sum(lengths)))
        np.random.seed(1)
        np.random.shuffle(indices)
        return [Subset(dataset, indices[offset - length:offset]) for offset, length in
                zip(_accumulate(lengths), lengths)]

        return all_data

    @classmethod
    def model_distillation(cls, model_access: LosstrajModelAccess, distillation_dataset: DataLoader,
                           auxiliary_info: LosstrajAuxiliaryInfo, teacher_type="target"):
        """
         Distill a model with the given distillation dataset, and save the distilled model at each epoch.
        :param model_access: the access to the teacher model
        :param distillation_dataset: the dataset used to obtain the soft labels from the target model and train the distilled model.
        :param auxiliary_info: the auxiliary information.
        :param teacher_type: the type of the teacher model. It can be "target" or "shadow".
        :return: None
        """
        distilled_model = model_access.get_model_architecture()
        distilled_model.to(auxiliary_info.device)
        distilled_model.train()
        optimizer = torch.optim.Adam(distilled_model.parameters(), lr=0.001)

        for epoch in tqdm(auxiliary_info.distillation_epochs):
            for i, data in enumerate(distillation_dataset):
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                teacher_pred = model_access.model(inputs)  # teacher model
                distilled_pred = distilled_model(inputs)  # student model
                loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(teacher_pred, dim=1),
                                                                 F.softmax(distilled_pred, dim=1))
                loss.backward()
                optimizer.step()

            # Calculate the accuracy for this batch
            distilled_model.eval()
            with torch.no_grad():
                correct_predictions = 0
                total_samples = 0
                for i, data in enumerate(distillation_dataset):
                    inputs, labels = data
                    inputs = inputs.to(auxiliary_info.device)
                    outputs = distilled_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                accuracy = correct_predictions / total_samples
                print('Epoch: {}, Accuracy: {}'.format(epoch, accuracy))

            # save the distilled model at the end of each epoch
            torch.save(distilled_model.state_dict(), os.path.join(auxiliary_info.distill_models_path, teacher_type,
                                                                  "distilled_model_ep" + epoch + ".pth"))

    @classmethod
    def train_shadow_model(cls, shadow_model, shadow_train_dataset, shadow_test_dataset,
                           auxiliary_info: LosstrajAuxiliaryInfo) -> LosstrajModelAccess:
        """
        Train the shadow model if the shadow model is not at auxiliary_info.shadow_model_path.
        :param shadow_model: the shadow model.
        :param shadow_train_dataset: the training dataset for the shadow model.
        :param shadow_test_dataset: the test dataset for the shadow model.
        :param auxiliary_info: the auxiliary information.
        :return: model access to the shadow model.
        """
        if os.path.exists(auxiliary_info.shadow_model_path):
            shadow_model.load_state_dict(torch.load(auxiliary_info.shadow_model_path))
            return LosstrajModelAccess(shadow_model, ModelAccessType.BLACK_BOX)

        shadow_model.to(auxiliary_info.device)
        shadow_model.train()
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)

        for epoch in tqdm(auxiliary_info.distillation_epochs):
            for i, data in enumerate(shadow_train_dataset):
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                outputs = shadow_model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

            # Calculate the accuracy for this batch
            shadow_model.eval()
            with torch.no_grad():
                correct_predictions = 0
                total_samples = 0
                for i, data in enumerate(shadow_test_dataset):
                    inputs, labels = data
                    inputs = inputs.to(auxiliary_info.device)
                    outputs = shadow_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                accuracy = correct_predictions / total_samples
                print('Epoch: {}, Accuracy (test dataset): {}'.format(epoch, accuracy))

        # save the model
        torch.save(shadow_model.state_dict(), auxiliary_info.shadow_model_path)
        return LosstrajModelAccess(shadow_model, ModelAccessType.BLACK_BOX)

    @classmethod
    def get_loss_trajectory(cls, data: DataLoader, model, auxiliary_info: LosstrajAuxiliaryInfo, model_type="target"):
        """
        Get the loss trajectory of the model specified by model_type.
        :param data: the dataset to obtain the loss trajectory.
        :param model: the model to load to.
        :param auxiliary_info: the auxiliary information.
        :param model_type: the type of the model. It can be "target" or "shadow".
        :return: the loss trajectory, a list of loss values at each epoch.
        """

        if model_type not in ["target", "shadow"]:
            raise ValueError("model_type should be either 'target' or 'shadow'!")

        # load each distilled model and record the loss trajectory
        loss_trajectory = []
        for epoch in tqdm(auxiliary_info.distillation_epochs):
            distilled_model = model
            distilled_model.to(auxiliary_info.device)
            distilled_model.load_state_dict(
                    torch.load(os.path.join(auxiliary_info.distill_models_path, model_type, "distilled_model_ep" + epoch + ".pth")))
            distilled_model.eval()
            with torch.no_grad():
                for i, data in enumerate(data):
                    inputs, labels = data
                    inputs = inputs.to(auxiliary_info.device)
                    outputs = distilled_model(inputs)
                    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                    loss_trajectory.append(loss.item())
        return loss_trajectory


    @classmethod
    def train_attack_model(cls, in_samples: list, out_samples: list, auxiliary_info: LosstrajAuxiliaryInfo, attack_model):
        """
        train the attack model with the in-sample and out-of-sample trajectories.
        :param in_samples: list of in-sample trajectories ([traj_epoch1, traj_epoch2, ..., traj_epochN, traj_un-distilled])
        :param out_samples: list of out-of-sample trajectories ([traj_epoch1, traj_epoch2, ..., traj_epochN, traj_un-distilled])
        :param auxiliary_info: the auxiliary information.
        :param attack_model: the attack model.
        :return: the attack model trained.
        """

        # prepare the dataset
        in_samples = np.array(in_samples)
        out_samples = np.array(out_samples)
        in_labels = np.ones(len(in_samples))
        out_labels = np.zeros(len(out_samples))
        all_samples = np.concatenate((in_samples, out_samples))
        all_labels = np.concatenate((in_labels, out_labels))
        all_samples = torch.tensor(all_samples, dtype=torch.float32)
        all_labels = torch.tensor(all_labels, dtype=torch.float32)
        attack_dataset = TensorDataset(all_samples, all_labels)

        # train the attack model
        attack_model.to(auxiliary_info.device)
        attack_model.train()
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(auxiliary_info.distillation_epochs):
            for i, data in attack_dataset:
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                labels = labels.to(auxiliary_info.device)
                optimizer.zero_grad()
                outputs = attack_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Calculate the accuracy for this batch
            attack_model.eval()
            with torch.no_grad():
                correct_predictions = 0
                total_samples = 0
                for i, data in attack_dataset:
                    inputs, labels = data
                    inputs = inputs.to(auxiliary_info.device)
                    labels = labels.to(auxiliary_info.device)
                    outputs = attack_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                accuracy = correct_predictions / total_samples
                print('Epoch: {}, Accuracy: {}'.format(epoch, accuracy))



        return attack_model



class LosstrajAttack(MiAttack):
    """
    Implementation of Losstraj attack.
    """

    def __init__(self, target_model_access: LosstrajModelAccess, auxiliary_info: LosstrajAuxiliaryInfo):
        """
        Initialize the attack.
        :param target_model_access: the target model access.
        :param auxiliary_info: the auxiliary information.
        """
        super().__init__(target_model_access, auxiliary_info)
        self.attack_model = None
        self.shadow_model = None
        self.distilled_target_model = None
        self.shadow_model_access = None
        self.shadow_test_dataset = None
        self.shadow_train_dataset = None
        self.distillation_test_dataset = None
        self.distillation_train_dataset = None
        self.auxiliary_info = auxiliary_info
        self.target_model_access = target_model_access

        # directories:
        for directory in [self.auxiliary_info.save_path, self.auxiliary_info.distill_models_path]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.prepared = False  # this flag indicates whether the attack is prepared

    def prepare(self, auxiliary_dataset):
        """
        Prepare the attack.
        """
        # determine the length of the distillation dataset and the shadow dataset
        distillation_train_len = int(len(auxiliary_dataset) * self.auxiliary_info.distillation_dataset_ratio)
        shadow_dataset_len = len(auxiliary_dataset) - distillation_train_len
        shadow_train_len = int(shadow_dataset_len * self.auxiliary_info.shadow_train_test_split_ratio)
        shadow_test_len = shadow_dataset_len - shadow_train_len

        self.distillation_train_dataset, self.shadow_train_dataset, self.shadow_test_dataset = LosstrajUtil.dataset_split(
            auxiliary_dataset, [distillation_train_len, shadow_train_len, shadow_test_len])

        # step 1: train shadow model, distill the shadow model and save the distilled models at each epoch
        self.shadow_model = self.target_model_access.get_model_architecture()
        self.shadow_model_access = LosstrajUtil.train_shadow_model(self.shadow_model, self.shadow_train_dataset, self.shadow_test_dataset,
                                        self.auxiliary_info)
        LosstrajUtil.model_distillation(self.shadow_model_access, self.distillation_train_dataset, self.auxiliary_info, teacher_type="shadow")

        # step 2: distill the target model and save the distilled models at each epoch
        LosstrajUtil.model_distillation(self.target_model_access, auxiliary_dataset, self.distillation_train_dataset, teacher_type="target")

        # step 3: obtain the loss trajectory of the shadow model and train the attack model
        shadow_train_loss_trajectory = LosstrajUtil.get_loss_trajectory(self.shadow_train_dataset, self.shadow_model, self.auxiliary_info, model_type="shadow")
        shadow_train_loss_trajectory.append(self.shadow_model_access.model(self.shadow_train_dataset).item())
        shadow_test_loss_trajectory = LosstrajUtil.get_loss_trajectory(self.shadow_test_dataset, self.shadow_model, self.auxiliary_info, model_type="shadow")
        shadow_test_loss_trajectory.append(self.shadow_model_access.model(self.shadow_test_dataset).item())

        self.attack_model = self.target_model_access.get_model_architecture()
        self.attack_model = LosstrajUtil.train_attack_model(shadow_train_loss_trajectory, shadow_test_loss_trajectory, self.auxiliary_info, self.attack_model)

        self.prepared = True

    def infer(self, dataset):
        """
        Infer the membership of the dataset.
        :param dataset: the dataset to infer.
        :return: the inferred membership.
        """
        if not self.prepared:
            raise ValueError("The attack is not prepared yet!")

        # obtain the loss trajectory of the target model
        target_loss_trajectory = LosstrajUtil.get_loss_trajectory(dataset, self.target_model_access.get_model_architecture(), self.auxiliary_info, model_type="target")
        target_loss_trajectory.append(self.target_model_access.model(dataset).item())

        # infer the membership
        target_pred = self.attack_model(target_loss_trajectory)

        return target_pred
