# This code implements the loss trajectory based membership inference attack, published in CCS 2022 "Membership
# Inference Attacks by Exploiting Loss Trajectory".
# The code is based on the code from
# https://github.com/DennisLiu2022/Membership-Inference-Attacks-by-Exploiting-Loss-Trajectory
import copy
import logging
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from typing import List, Optional, Union

from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import get_num_classes, dataset_split
from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils


class AttackMLP(torch.nn.Module):
    # default model for the attack
    def __init__(self, dim_in):
        super(AttackMLP, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


class LosstrajAuxiliaryInfo(AuxiliaryInfo):
    """
    The auxiliary information for the loss trajectory based membership inference attack.
    """

    def __init__(self, config, attack_model=AttackMLP):
        """
        Initialize the auxiliary information.
        :param config: the loss trajectory.
        :param attack_model: the attack model.
        """
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.get('seed', 0)
        self.batch_size = config.get('batch_size', 128)
        self.num_workers = config.get('num_workers', 2)
        self.distillation_epochs = config.get('distillation_epochs', 100)

        # directories:
        self.save_path = config.get('save_path', './losstraj_files')
        self.distill_models_path = self.save_path + '/distill_models'
        self.shadow_model_path = self.save_path + '/shadow_model.pkl'
        self.shadow_losstraj_path = self.save_path + '/shadow_losstraj'

        # dataset length: it should be given as the ratio of the training dataset length w.r.t. the whole auxiliary dataset
        self.distillation_dataset_ratio = config.get('distillation_dataset_ratio', 0.5)
        self.shadow_dataset_ratio = 1 - self.distillation_dataset_ratio
        # train/test split ratio of the shadow dataset
        self.shadow_train_test_split_ratio = config.get('shadow_train_test_split_ratio', 0.5)
        self.num_classes = config.get('num_classes', 10)

        # training parameters
        self.lr = config.get('lr', 0.1)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.attack_model = attack_model

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)

        if self.log_path is not None:
            self.logger = logging.getLogger('loss_traj_logger')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/loss_traj.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


class LosstrajModelAccess(ModelAccess):
    """
    The model access for the loss trajectory based membership inference attack.
    """

    def __init__(self, model, untrained_model, model_type: ModelAccessType = ModelAccessType.BLACK_BOX):
        """
        Initialize the model access.
        :param model: the target model.
        :param model_type: the type of the target model.
        """
        super().__init__(model, untrained_model, model_type)


class LosstrajUtil(MIAUtils):
    @classmethod
    def model_distillation(cls, teacher_model_access: LosstrajModelAccess, distillation_dataset: TensorDataset,
                           auxiliary_info: LosstrajAuxiliaryInfo, teacher_type="target"):
        """
         Distill a model with the given distillation dataset, and save the distilled model at each epoch.
        :param teacher_model_access: the access to the teacher model
        :param distillation_dataset: the dataset used to obtain the soft labels from the target model and train the distilled model.
        :param auxiliary_info: the auxiliary information.
        :param teacher_type: the type of the teacher model. It can be "target" or "shadow".
        :return: None
        """
        print(
            f"getting distilled model with teacher type: {teacher_type} on distillation dataset of len: {len(distillation_dataset)}")
        if auxiliary_info.log_path is not None:
            auxiliary_info.logger.info(
                f"getting distilled model with teacher type: {teacher_type} on distillation dataset of len: {len(distillation_dataset)}")
        if not os.path.exists(os.path.join(auxiliary_info.distill_models_path, teacher_type)):
            os.makedirs(os.path.join(auxiliary_info.distill_models_path, teacher_type))
        elif len(os.listdir(
                os.path.join(auxiliary_info.distill_models_path, teacher_type))) >= auxiliary_info.distillation_epochs:
            return  # if the distilled models are already saved, return

        distilled_model = copy.deepcopy(teacher_model_access.get_untrained_model())
        distilled_model.to(auxiliary_info.device)
        teacher_model_access.to_device(auxiliary_info.device)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, distilled_model.parameters()),
                                    lr=auxiliary_info.lr, momentum=auxiliary_info.momentum,
                                    weight_decay=auxiliary_info.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, auxiliary_info.distillation_epochs)

        distill_train_loader = DataLoader(distillation_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                          num_workers=auxiliary_info.num_workers)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            distilled_model.train()
            teacher_model_access.eval()
            train_loss = 0
            for i, data in enumerate(distill_train_loader):
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                optimizer.zero_grad()
                teacher_pred = teacher_model_access.get_signal(inputs)  # teacher model
                distilled_pred = distilled_model(inputs)  # student model
                loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(distilled_pred, dim=1),
                                                                 F.softmax(teacher_pred, dim=1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # Calculate the accuracy for this batch
            if epoch % 20 == 0 or epoch == auxiliary_info.distillation_epochs - 1:
                distilled_model.eval()
                teacher_model_access.eval()
                with torch.no_grad():
                    student_correct_count = 0
                    teacher_correct_count = 0
                    total_samples = 0
                    for i, data in enumerate(distill_train_loader):
                        inputs, labels = data
                        inputs = inputs.to(auxiliary_info.device)
                        labels = labels.to(auxiliary_info.device)
                        student_outputs = distilled_model(inputs)
                        teacher_outputs = teacher_model_access.get_signal(inputs)
                        _, student_predicted = torch.max(student_outputs, 1)
                        _, teacher_predicted = torch.max(teacher_outputs, 1)
                        student_correct_count += (student_predicted == labels).sum().item()
                        teacher_correct_count += (teacher_predicted == labels).sum().item()
                        total_samples += labels.size(0)

                    # Calculate accuracy after iterating through the entire dataset
                    student_acc = (student_correct_count / total_samples) * 100
                    teacher_acc = (teacher_correct_count / total_samples) * 100
                    print(
                        f"Epoch {epoch + 1}, student_train acc: {student_acc:.2f}%, teacher acc: {teacher_acc:.2f}%, "
                        f"Loss: {train_loss:.2f}, lr: {scheduler.get_last_lr()[0]:.4f}")
                    if auxiliary_info.log_path is not None:
                        auxiliary_info.logger.info(
                            f"Epoch {epoch + 1}, student_train acc: {student_acc:.2f}%, teacher acc: {teacher_acc:.2f}%, "
                            f"Loss: {train_loss:.2f}, lr: {scheduler.get_last_lr()[0]:.4f}")

            # save the model
            torch.save(distilled_model.state_dict(),
                       os.path.join(auxiliary_info.distill_models_path, teacher_type,
                                    "distilled_model_ep" + str(epoch) + ".pkl"))

    @classmethod
    def label_to_distribution(cls, label, num_classes):
        """
        Convert a label to a one-hot distribution.
        :param label:
        :param num_classes:
        :return:
        """
        identity_matrix = torch.eye(num_classes, dtype=torch.float)
        distribution = identity_matrix[label]
        return distribution

    @classmethod
    def to_categorical(cls, labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
        """
        Convert an array of labels to binary class matrix.

        :param labels: An array of integer labels of shape `(nb_samples,)`.
        :param nb_classes: The number of classes (possible labels).
        :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
        """
        labels = np.array(labels, dtype=np.int32)
        if nb_classes is None:
            nb_classes = np.max(labels) + 1
        categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
        categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1

        return categorical

    @classmethod
    def check_and_transform_label_format(cls,
                                         labels: np.ndarray, nb_classes: Optional[int] = None,
                                         return_one_hot: bool = True
                                         ) -> np.ndarray:
        """
        Check label format and transform to one-hot-encoded labels if necessary

        :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
        :param nb_classes: The number of classes.
        :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
        :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
        """
        if labels is not None:
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                if not return_one_hot:
                    labels = np.argmax(labels, axis=1)
            elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes > 2:
                labels = np.squeeze(labels)
                if return_one_hot:
                    labels = cls.to_categorical(labels, nb_classes)
            elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes == 2:
                pass
            elif len(labels.shape) == 1:
                if return_one_hot:
                    if nb_classes == 2:
                        labels = np.expand_dims(labels, axis=1)
                    else:
                        labels = cls.to_categorical(labels, nb_classes)
            else:
                raise ValueError(
                    "Shape of labels not recognised."
                    "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
                )

        return labels

    @classmethod
    def train_shadow_model(cls, shadow_model, shadow_train_dataset, shadow_test_dataset,
                           auxiliary_info: LosstrajAuxiliaryInfo, seed) -> LosstrajModelAccess:
        """
        Train the shadow model if the shadow model is not at auxiliary_info.shadow_model_path.
        :param shadow_model: the shadow model.
        :param shadow_train_dataset: the training dataset for the shadow model.
        :param shadow_test_dataset: the test dataset for the shadow model.
        :param auxiliary_info: the auxiliary information.
        :return: model access to the shadow model.
        """

        try:
            set_seed(seed)
            shadow_model.initialize_weights()
        except:
            raise NotImplementedError("the model doesn't have .initialize_weights method")


        print(
            f"obtaining shadow model with trainset len: {len(shadow_train_dataset)} and testset len: {len(shadow_test_dataset)}")
        if auxiliary_info.log_path is not None:
            auxiliary_info.logger.info(
                f"obtaining shadow model with trainset len: {len(shadow_train_dataset)} and testset len: {len(shadow_test_dataset)}")
        untrained_shadow_model = copy.deepcopy(shadow_model)

        if os.path.exists(auxiliary_info.shadow_model_path):
            shadow_model.load_state_dict(torch.load(auxiliary_info.shadow_model_path))
            return LosstrajModelAccess(shadow_model, untrained_shadow_model, ModelAccessType.BLACK_BOX)

        shadow_model.to(auxiliary_info.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, shadow_model.parameters()), lr=auxiliary_info.lr,
                                    momentum=auxiliary_info.momentum,
                                    weight_decay=auxiliary_info.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, auxiliary_info.distillation_epochs)

        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                         num_workers=auxiliary_info.num_workers)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=auxiliary_info.batch_size, shuffle=False,
                                        num_workers=auxiliary_info.num_workers)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            shadow_model.train()
            train_loss = 0
            for i, data in enumerate(shadow_train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(auxiliary_info.device), labels.to(auxiliary_info.device)
                optimizer.zero_grad()
                outputs = shadow_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # Calculate the accuracy for this batch
            if epoch % 20 == 0 or epoch == auxiliary_info.distillation_epochs - 1:
                shadow_model.eval()
                with torch.no_grad():
                    test_correct_predictions = 0
                    total_samples = 0
                    for i, data in enumerate(shadow_test_loader):
                        inputs, labels = data
                        inputs, labels = inputs.to(auxiliary_info.device), labels.to(auxiliary_info.device)
                        outputs = shadow_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        test_correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    test_accuracy = test_correct_predictions / total_samples

                    train_correct_predictions = 0
                    total_samples = 0
                    for i, data in enumerate(shadow_train_loader):
                        inputs, labels = data
                        inputs, labels = inputs.to(auxiliary_info.device), labels.to(auxiliary_info.device)
                        outputs = shadow_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        train_correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    train_accuracy = train_correct_predictions / total_samples
                print(
                    f"Epoch {epoch}, train_acc: {train_accuracy:.2f}, test_acc: {test_accuracy:.2f}%, Loss: {train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
                if auxiliary_info.log_path is not None:
                    auxiliary_info.logger.info(
                        f"Epoch {epoch}, train_acc: {train_accuracy:.2f}, test_acc: {test_accuracy:.2f}%, Loss: {train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
        # save the model
        torch.save(shadow_model.state_dict(), auxiliary_info.shadow_model_path)
        return LosstrajModelAccess(shadow_model, untrained_shadow_model, ModelAccessType.BLACK_BOX)

    @classmethod
    def get_loss_trajectory(cls, data, model, auxiliary_info: LosstrajAuxiliaryInfo, model_type="target") -> np.ndarray:
        """
        Get the loss trajectory of the model specified by model_type.
        :param data: the dataset to obtain the loss trajectory.
        :param model: the model to load to.
        :param auxiliary_info: the auxiliary information.
        :param model_type: the type of the model. It can be "target" or "shadow".
        :return: the loss trajectory, where each row is the loss trajectory of a sample.
        """
        loss_array = np.array([])
        loss_trajectory = np.array([])

        if model_type not in ["target", "shadow"]:
            raise ValueError("model_type should be either 'target' or 'shadow'!")

        # create loader for the dataset
        data_loader = DataLoader(data, batch_size=auxiliary_info.batch_size, shuffle=False)

        # load each distilled model and record the loss trajectory
        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            distilled_model = model
            distilled_model.to(auxiliary_info.device)
            distilled_model.load_state_dict(
                torch.load(os.path.join(auxiliary_info.distill_models_path, model_type,
                                        "distilled_model_ep" + str(epoch) + ".pkl")))
            distilled_model.eval()
            with torch.no_grad():
                iter_count = 0
                for i, data in enumerate(data_loader):
                    inputs, labels = data
                    inputs = inputs.to(auxiliary_info.device)
                    labels = labels.to(auxiliary_info.device)
                    outputs = distilled_model(inputs)
                    loss = [nn.functional.cross_entropy(output.unsqueeze(0), label.unsqueeze(0)) for (output, label) in
                            zip(outputs, labels)]
                    loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss])
                    loss = loss.reshape(-1, 1)
                    loss_array = np.concatenate([loss_array, loss], axis=0) if iter_count > 0 else loss
                    iter_count += 1

            loss_trajectory = loss_array if epoch == 0 else np.concatenate([loss_trajectory, loss_array], axis=1)

        return loss_trajectory

    @classmethod
    def get_loss(cls, data, model_access: LosstrajModelAccess, auxiliary_info: LosstrajAuxiliaryInfo) -> np.ndarray:
        """
        Get the loss of model on given data.
        :param data: the dataset to obtain the loss.
        :param model_access: the model access.
        :param auxiliary_info: the auxiliary information.
        :return:
        """

        # create loader for the dataset
        data_loader = DataLoader(data, batch_size=auxiliary_info.batch_size, shuffle=False)

        model_access.to_device(auxiliary_info.device)
        model_access.model.eval()
        with torch.no_grad():
            loss_array = []
            iter_count = 0
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                labels = labels.to(auxiliary_info.device)
                outputs = model_access.get_signal(inputs)
                loss = [nn.functional.cross_entropy(output.unsqueeze(0), label.unsqueeze(0)) for (output, label) in
                        zip(outputs, labels)]
                loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
                loss_array = np.concatenate([loss_array, loss], axis=0) if iter_count > 0 else loss
                iter_count += 1

        return loss_array

    @classmethod
    def train_attack_model(cls, attack_dataset, auxiliary_info: LosstrajAuxiliaryInfo, attack_model, test_ratio=0):
        """
        train the attack model with the in-sample and out-of-sample trajectories.
        :param attack_dataset: the dataset used to train attack model
        :param auxiliary_info: the auxiliary information.
        :param attack_model: the attack model.
        :param test_ratio: the ratio of the test set, 0 means no test set.

        :return: the attack model trained.
        """

        if os.path.exists(os.path.join(auxiliary_info.save_path, "attack_model.pkl")):
            attack_model.load_state_dict(torch.load(os.path.join(auxiliary_info.save_path, "attack_model.pkl")))
            attack_model.to(auxiliary_info.device)
            return attack_model

        # split the dataset to train set and test set:
        attack_model.to(auxiliary_info.device)
        trainset_len = int(len(attack_dataset) * test_ratio)
        trainset, testset = dataset_split(attack_dataset,
                                          [trainset_len, len(attack_dataset) - trainset_len]) if test_ratio > 0 else (
            attack_dataset, None)

        train_loader = DataLoader(trainset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                  num_workers=auxiliary_info.num_workers)
        if test_ratio > 0:
            test_loader = DataLoader(testset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                     num_workers=auxiliary_info.num_workers)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, attack_model.parameters()), lr=0.01,
                                    momentum=auxiliary_info.momentum, weight_decay=auxiliary_info.weight_decay)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            attack_model.train()
            train_loss = 0
            for data in train_loader:
                losstraj, membership = data
                losstraj, membership = losstraj.to(auxiliary_info.device), membership.to(auxiliary_info.device)
                optimizer.zero_grad()
                output = attack_model(losstraj)
                loss = criterion(output, membership)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if epoch % 20 == 0 or epoch == auxiliary_info.distillation_epochs - 1:
                attack_model.eval()
                train_correct_predictions = 0
                total_samples = 0
                for i, data in enumerate(train_loader):
                    losstraj, membership = data
                    losstraj, membership = losstraj.to(auxiliary_info.device), membership.to(auxiliary_info.device)
                    outputs = attack_model(losstraj)
                    _, predicted = torch.max(outputs, 1)
                    train_correct_predictions += (predicted == membership).sum().item()
                    total_samples += membership.size(0)
                train_accuracy = train_correct_predictions / total_samples

                test_correct_predictions = 0
                total_samples = 0
                if test_ratio == 0:
                    continue
                for i, data in enumerate(test_loader):
                    losstraj, membership = data
                    losstraj, membership = losstraj.to(auxiliary_info.device), membership.to(auxiliary_info.device)
                    outputs = attack_model(losstraj)
                    _, predicted = torch.max(outputs, 1)
                    test_correct_predictions += (predicted == membership).sum().item()
                    total_samples += membership.size(0)
                test_accuracy = test_correct_predictions / total_samples
                if test_ratio > 0:
                    print(
                        f"Epoch {epoch}, train_acc: {train_accuracy:.2f}, test_acc: {test_accuracy:.2f}%, Loss: {train_loss:.4f}")
                    if auxiliary_info.log_path is not None:
                        auxiliary_info.logger.info(
                            f"Epoch {epoch}, train_acc: {train_accuracy:.2f}, test_acc: {test_accuracy:.2f}%, Loss: {train_loss:.4f}")
                else:
                    print(f"Epoch {epoch}, train_acc: {train_accuracy:.2f}, Loss: {train_loss:.4f}")
                    if auxiliary_info.log_path is not None:
                        auxiliary_info.logger.info(
                            f"Epoch {epoch}, train_acc: {train_accuracy:.2f}, Loss: {train_loss:.4f}")

        # save the model
        torch.save(attack_model.state_dict(), os.path.join(auxiliary_info.save_path, "attack_model.pkl"))
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
        self.aux_info = auxiliary_info
        self.target_model_access = target_model_access
        self.num_classes = auxiliary_info.num_classes

        # directories:
        for directory in [self.aux_info.save_path, self.aux_info.distill_models_path]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.prepared = False  # this flag indicates whether the attack is prepared

    def prepare(self, auxiliary_dataset):
        """
        Prepare the attack.
        :param auxiliary_dataset: the auxiliary dataset.
        """

        if self.prepared:
            print("the attack is already prepared!")
            return

        if get_num_classes(auxiliary_dataset) != self.aux_info.num_classes:
            raise ValueError(
                "The number of classes in the auxiliary dataset does not match the number of classes in the auxiliary information!")

        attack_model = self.aux_info.attack_model

        # saving config
        with open(self.aux_info.save_path + '/losstraj_attack_config.json', 'w') as f:
            json.dump(self.aux_info.save_config_to_dict(), f)

        LosstrajUtil.log(self.aux_info, "Start preparing the attack...", print_flag=True)

        # set the seed
        set_seed(self.aux_info.seed)
        print(f"LOSSTRAJ: setting seed to {self.aux_info.seed}")
        if self.aux_info.log_path is not None:
            self.aux_info.logger.info(f"LOSSTRAJ: setting seed to {self.aux_info.seed}")
        # determine the length of the distillation dataset and the shadow dataset
        distillation_train_len = int(len(auxiliary_dataset) * self.aux_info.distillation_dataset_ratio)
        shadow_dataset_len = len(auxiliary_dataset) - distillation_train_len
        shadow_train_len = int(shadow_dataset_len * self.aux_info.shadow_train_test_split_ratio)
        shadow_test_len = shadow_dataset_len - shadow_train_len

        self.distillation_train_dataset, self.shadow_train_dataset, self.shadow_test_dataset = dataset_split(
            auxiliary_dataset, [distillation_train_len, shadow_train_len, shadow_test_len])

        # step 1: train shadow model, distill the shadow model and save the distilled models at each epoch
        print("PREPARE: Training shadow model...")
        if self.aux_info.log_path is not None:
            self.aux_info.logger.info("PREPARE: Training shadow model...")

        self.shadow_model = self.target_model_access.get_untrained_model()
        self.shadow_model_access = LosstrajUtil.train_shadow_model(self.shadow_model, self.shadow_train_dataset,
                                                                   self.shadow_test_dataset,
                                                                   self.aux_info, self.aux_info.seed)
        print("PREPARE: Distilling shadow model...")
        if self.aux_info.log_path is not None:
            self.aux_info.logger.info("PREPARE: Distilling shadow model...")
        LosstrajUtil.model_distillation(self.shadow_model_access, self.distillation_train_dataset, self.aux_info,
                                        teacher_type="shadow")

        # step 2: distill the target model and save the distilled models at each epoch
        print("PREPARE: Distilling target model...")
        if self.aux_info.log_path is not None:
            self.aux_info.logger.info("PREPARE: Distilling target model...")
        LosstrajUtil.model_distillation(self.target_model_access, self.distillation_train_dataset, self.aux_info,
                                        teacher_type="target")

        # step 3: obtain the loss trajectory of the shadow model and train the attack model
        print("PREPARE: Obtaining loss trajectory of the shadow model...")
        if self.aux_info.log_path is not None:
            self.aux_info.logger.info("PREPARE: Obtaining loss trajectory of the shadow model...")
        if not os.path.exists(self.aux_info.shadow_losstraj_path):
            os.makedirs(self.aux_info.shadow_losstraj_path)

        if not os.path.exists(self.aux_info.shadow_losstraj_path + '/shadow_train_loss_traj.npy'):
            shadow_train_loss_trajectory = LosstrajUtil.get_loss_trajectory(
                self.shadow_train_dataset,
                self.target_model_access.get_untrained_model(),
                self.aux_info,
                model_type="shadow")
            original_shadow_traj = LosstrajUtil.get_loss(self.shadow_train_dataset, self.shadow_model_access,
                                                         self.aux_info).reshape(-1, 1)
            shadow_train_loss_trajectory = np.concatenate([shadow_train_loss_trajectory, original_shadow_traj], axis=1)
            np.save(self.aux_info.shadow_losstraj_path + '/shadow_train_loss_traj.npy',
                    np.array(shadow_train_loss_trajectory))
        else:
            shadow_train_loss_trajectory = np.load(
                self.aux_info.shadow_losstraj_path + '/shadow_train_loss_traj.npy', allow_pickle=True)

        if not os.path.exists(self.aux_info.shadow_losstraj_path + '/shadow_test_loss_traj.npy'):
            shadow_test_loss_trajectory = LosstrajUtil.get_loss_trajectory(
                self.shadow_test_dataset,
                self.shadow_model,
                self.aux_info,
                model_type="shadow")
            original_shadow_traj = LosstrajUtil.get_loss(self.shadow_test_dataset, self.shadow_model_access,
                                                         self.aux_info).reshape(-1, 1)
            shadow_test_loss_trajectory = np.concatenate([shadow_test_loss_trajectory, original_shadow_traj], axis=1)
            np.save(self.aux_info.shadow_losstraj_path + '/shadow_test_loss_traj.npy',
                    np.array(shadow_test_loss_trajectory))
        else:
            shadow_test_loss_trajectory = np.load(
                self.aux_info.shadow_losstraj_path + '/shadow_test_loss_traj.npy', allow_pickle=True)

        # create the dataset for the attack model
        train_membership = np.ones(len(shadow_train_loss_trajectory))
        test_membership = np.zeros(len(shadow_test_loss_trajectory))
        attack_membership = np.concatenate([train_membership, test_membership])
        traj_data = np.concatenate([shadow_train_loss_trajectory, shadow_test_loss_trajectory])
        traj_data = torch.tensor(traj_data, dtype=torch.float32)
        attack_membership = torch.tensor(attack_membership, dtype=torch.long)
        attack_dataset = TensorDataset(traj_data, attack_membership)

        print("PREPARE: Training attack model...")
        if self.aux_info.log_path is not None:
            self.aux_info.logger.info("PREPARE: Training attack model...")
        self.attack_model = attack_model(self.aux_info.distillation_epochs + 1)
        self.attack_model = LosstrajUtil.train_attack_model(attack_dataset,
                                                            self.aux_info, self.attack_model)

        self.prepared = True
        LosstrajUtil.log(self.aux_info, "Finish preparing the attack...", print_flag=True)

    def infer(self, dataset) -> np.ndarray:
        """
        Infer the membership of the dataset.
        :param dataset: the dataset to infer.
        :return: the inferred membership of shape
        """
        if not self.prepared:
            raise ValueError("The attack is not prepared yet!")

        set_seed(self.aux_info.seed)

        LosstrajUtil.log(self.aux_info, "Start membership inference...", print_flag=True)

        # obtain the loss trajectory of the target model
        print("INFER: Obtaining loss trajectory of the target model...")
        target_loss_trajectory = LosstrajUtil.get_loss_trajectory(dataset,
                                                                  self.target_model_access.get_untrained_model(),
                                                                  self.aux_info, model_type="target")

        original_target_traj = LosstrajUtil.get_loss(dataset, self.target_model_access, self.aux_info).reshape(-1,
                                                                                                                     1)
        target_loss_trajectory = np.concatenate([target_loss_trajectory, original_target_traj], axis=1)

        # infer the membership
        data_to_infer = np.array(target_loss_trajectory)
        data_to_infer = torch.tensor(data_to_infer, dtype=torch.float32)
        target_pred = self.attack_model(data_to_infer.to(self.aux_info.device))

        target_pred = target_pred.detach().cpu().numpy()
        target_pred = -np.transpose(target_pred)[0]

        LosstrajUtil.log(self.aux_info, "Finish membership inference...", print_flag=True)
        return target_pred
