# This code implements the loss trajectory based membership inference attack, published in CCS 2022 "Membership
# Inference Attacks by Exploiting Loss Trajectory".
# The code is based on the code from
# https://github.com/DennisLiu2022/Membership-Inference-Attacks-by-Exploiting-Loss-Trajectory
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from typing import List, Optional, Union

from mia.utils.set_seed import set_seed
from mia.utils.dataset_utils import get_num_classes, dataset_split
from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack



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
        self.batch_size = config.get('batch_size', 1000)
        self.num_workers = config.get('num_workers', 2)
        self.distillation_epochs = config.get('distillation_epochs', 100)

        # directories:
        self.save_path = config.get('save_path', './losstraj_files')
        self.distill_models_path = self.save_path + '/distill_models'
        self.shadow_model_path = self.save_path + '/shadow_model.pth'
        self.shadow_losstraj_path = self.save_path + '/shadow_losstraj'

        # dataset length: it should be given as the ratio of the training dataset length w.r.t. the whole auxiliary dataset
        self.distillation_dataset_ratio = config.get('distillation_dataset_ratio', 0.6)
        self.shadow_dataset_ratio = 1 - self.distillation_dataset_ratio
        # train/test split ratio of the shadow dataset
        self.shadow_train_test_split_ratio = config.get('shadow_train_test_split_ratio', 0.5)
        self.num_classes = config.get('num_classes', 10)

        # training parameters
        self.lr = config.get('lr', 0.1)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 0.0001)

        self.attack_model = attack_model


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


class LosstrajUtil:
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
        print(f"getting distilled model with teacher type: {teacher_type} on distillation dataset of len: {len(distillation_dataset)}")
        if os.path.exists(os.path.join(auxiliary_info.distill_models_path, teacher_type)) == False:
            os.makedirs(os.path.join(auxiliary_info.distill_models_path, teacher_type))
        elif len(os.listdir(
                os.path.join(auxiliary_info.distill_models_path, teacher_type))) >= auxiliary_info.distillation_epochs:
            return  # if the distilled models are already saved, return

        distilled_model = copy.deepcopy(teacher_model_access.get_untrained_model())
        distilled_model.to(auxiliary_info.device)
        teacher_model_access.to_device(auxiliary_info.device)
        distilled_model.train()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, distilled_model.parameters()), lr=auxiliary_info.lr, momentum=auxiliary_info.momentum,
                                    weight_decay=auxiliary_info.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, auxiliary_info.distillation_epochs)

        distill_train_loader = DataLoader(distillation_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                          num_workers=auxiliary_info.num_workers)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
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
            scheduler.step()

            # Calculate the accuracy for this batch
            distilled_model.eval()
            if epoch % 20 == 0 or epoch == auxiliary_info.distillation_epochs - 1:
                with torch.no_grad():
                    train_correct_count = 0
                    total_samples = 0
                    for i, data in enumerate(distill_train_loader):
                        inputs, labels = data
                        inputs = inputs.to(auxiliary_info.device)
                        labels = labels.to(auxiliary_info.device)
                        outputs = distilled_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        train_correct_count += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    train_acc = train_correct_count / total_samples
                    print(f"Epoch {epoch + 1}, train acc: {train_acc:.2f}%, Loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")

            # save the distilled model at the end of each epoch
            torch.save(distilled_model.state_dict(), os.path.join(auxiliary_info.distill_models_path, teacher_type,
                                                                  "distilled_model_ep" + str(epoch) + ".pth"))

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
                           auxiliary_info: LosstrajAuxiliaryInfo) -> LosstrajModelAccess:
        """
        Train the shadow model if the shadow model is not at auxiliary_info.shadow_model_path.
        :param shadow_model: the shadow model.
        :param shadow_train_dataset: the training dataset for the shadow model.
        :param shadow_test_dataset: the test dataset for the shadow model.
        :param auxiliary_info: the auxiliary information.
        :return: model access to the shadow model.
        """

        print(f"obtaining shadow model with trainset len: {len(shadow_train_dataset)} and testset len: {len(shadow_test_dataset)}")

        untrained_shadow_model = copy.deepcopy(shadow_model)

        if os.path.exists(auxiliary_info.shadow_model_path):
            shadow_model.load_state_dict(torch.load(auxiliary_info.shadow_model_path))
            return LosstrajModelAccess(shadow_model, untrained_shadow_model, ModelAccessType.BLACK_BOX)

        shadow_model.to(auxiliary_info.device)
        shadow_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, shadow_model.parameters()), lr=auxiliary_info.lr, momentum=auxiliary_info.momentum,
                                    weight_decay=auxiliary_info.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, auxiliary_info.distillation_epochs)

        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                         num_workers=auxiliary_info.num_workers)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=auxiliary_info.batch_size, shuffle=False,
                                        num_workers=auxiliary_info.num_workers)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            for i, data in enumerate(shadow_train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(auxiliary_info.device), labels.to(auxiliary_info.device)
                optimizer.zero_grad()
                outputs = shadow_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
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
            f"Epoch {epoch}, train_acc: {train_accuracy:.2f}, test_acc: {test_accuracy:.2f}%, Loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")

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
                                        "distilled_model_ep" + str(epoch) + ".pth")))
            distilled_model.eval()
            with torch.no_grad():
                iter_count = 0
                for i, data in enumerate(data_loader):
                    inputs, labels = data
                    inputs = inputs.to(auxiliary_info.device)
                    labels = labels.to(auxiliary_info.device)
                    outputs = distilled_model(inputs)
                    loss = [nn.functional.cross_entropy(output.unsqueeze(0), label.unsqueeze(0)) for (output, label) in zip(outputs, labels)]
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
                outputs = model_access.model(inputs)
                loss = [nn.functional.cross_entropy(output.unsqueeze(0), label.unsqueeze(0)) for (output, label) in zip(outputs, labels)]
                loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
                loss_array = np.concatenate([loss_array, loss], axis=1) if iter_count > 0 else loss
                iter_count += 1

        return loss_array

    @classmethod
    def train_attack_model(cls, in_samples: np.ndarray, out_samples: np.ndarray, auxiliary_info: LosstrajAuxiliaryInfo,
                           attack_model, num_classes: int):
        """
        train the attack model with the in-sample and out-of-sample trajectories.
        :param in_samples: arr of in-sample trajectories ([traj_epoch1 traj_epoch2 ... traj_epochN traj_un-distilled])
        :param out_samples: arr of out-of-sample trajectories ([traj_epoch1 traj_epoch2 ..., traj_epochN traj_un-distilled])
        :param auxiliary_info: the auxiliary information.
        :param attack_model: the attack model.
        :param num_classes: the number of classes.
        :return: the attack model trained.
        """

        # prepare the dataset
        in_labels = np.ones(len(in_samples))
        out_labels = np.zeros(len(out_samples))
        all_samples = np.concatenate((in_samples, out_samples))
        all_labels = np.concatenate((in_labels, out_labels))
        all_samples = torch.tensor(all_samples, dtype=torch.float32)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        attack_dataset = TensorDataset(all_samples, all_labels)

        # train the attack model
        attack_model.to(auxiliary_info.device)
        attack_model.train()
        optimizer = torch.optim.SGD(attack_model.parameters(), lr=auxiliary_info.lr, momentum=auxiliary_info.momentum,
                                    weight_decay=auxiliary_info.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        # no scheduler used in the losstraj original code
        # scheduler = CosineAnnealingLR(optimizer, auxiliary_info.distillation_epochs)

        attack_loader = DataLoader(attack_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                   num_workers=auxiliary_info.num_workers)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            for data in attack_loader:
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                labels = labels.to(auxiliary_info.device)
                optimizer.zero_grad()
                outputs = attack_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if epoch % 20 == 0 or epoch == auxiliary_info.distillation_epochs - 1:
                attack_model.eval()
                with torch.no_grad():
                    correct_predictions = 0
                    total_samples = 0
                    for data in attack_loader:
                        inputs, labels = data
                        inputs = inputs.to(auxiliary_info.device)
                        labels = labels.to(auxiliary_info.device)
                        outputs = attack_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    accuracy = correct_predictions / total_samples
                    print(f"Epoch {epoch + 1}, train_acc: {accuracy:.2f}%, Loss: {loss.item():.4f}")
            # scheduler.step()
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
        self.num_classes = auxiliary_info.num_classes

        # directories:
        for directory in [self.auxiliary_info.save_path, self.auxiliary_info.distill_models_path]:
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

        if get_num_classes(auxiliary_dataset) != self.auxiliary_info.num_classes:
            raise ValueError(
                "The number of classes in the auxiliary dataset does not match the number of classes in the auxiliary information!")

        attack_model = self.auxiliary_info.attack_model
        # set the seed
        set_seed(self.auxiliary_info.seed)
        # determine the length of the distillation dataset and the shadow dataset
        distillation_train_len = int(len(auxiliary_dataset) * self.auxiliary_info.distillation_dataset_ratio)
        shadow_dataset_len = len(auxiliary_dataset) - distillation_train_len
        shadow_train_len = int(shadow_dataset_len * self.auxiliary_info.shadow_train_test_split_ratio)
        shadow_test_len = shadow_dataset_len - shadow_train_len

        self.distillation_train_dataset, self.shadow_train_dataset, self.shadow_test_dataset = dataset_split(
            auxiliary_dataset, [distillation_train_len, shadow_train_len, shadow_test_len])

        # step 1: train shadow model, distill the shadow model and save the distilled models at each epoch
        print("PREPARE: Training shadow model...")
        self.shadow_model = self.target_model_access.get_untrained_model()
        self.shadow_model_access = LosstrajUtil.train_shadow_model(self.shadow_model, self.shadow_train_dataset,
                                                                   self.shadow_test_dataset,
                                                                   self.auxiliary_info)
        print("PREPARE: Distilling shadow model...")
        LosstrajUtil.model_distillation(self.shadow_model_access, self.distillation_train_dataset, self.auxiliary_info,
                                        teacher_type="shadow")

        # step 2: distill the target model and save the distilled models at each epoch
        print("PREPARE: Distilling target model...")
        LosstrajUtil.model_distillation(self.target_model_access, self.distillation_train_dataset, self.auxiliary_info,
                                        teacher_type="target")

        # step 3: obtain the loss trajectory of the shadow model and train the attack model
        print("PREPARE: Obtaining loss trajectory of the shadow model...")
        if os.path.exists(self.auxiliary_info.shadow_losstraj_path) == False:
            os.makedirs(self.auxiliary_info.shadow_losstraj_path)

        if not os.path.exists(self.auxiliary_info.shadow_losstraj_path + '/shadow_train_loss_traj.npy'):
            shadow_train_loss_trajectory = LosstrajUtil.get_loss_trajectory(self.shadow_train_dataset,
                                                                            self.shadow_model,
                                                                            self.auxiliary_info, model_type="shadow")
            original_shadow_traj = LosstrajUtil.get_loss(self.shadow_train_dataset, self.shadow_model_access,
                                                         self.auxiliary_info).reshape(-1, 1)
            shadow_train_loss_trajectory = np.concatenate([original_shadow_traj, shadow_train_loss_trajectory], axis=1)
            np.save(self.auxiliary_info.shadow_losstraj_path + '/shadow_train_loss_traj.npy',
                    np.array(shadow_train_loss_trajectory))
        else:
            shadow_train_loss_trajectory = np.load(
                self.auxiliary_info.shadow_losstraj_path + '/shadow_train_loss_traj.npy', allow_pickle=True)

        if not os.path.exists(self.auxiliary_info.shadow_losstraj_path + '/shadow_test_loss_traj.npy'):
            shadow_test_loss_trajectory = LosstrajUtil.get_loss_trajectory(self.shadow_test_dataset, self.shadow_model,
                                                                           self.auxiliary_info, model_type="shadow")
            original_shadow_traj = LosstrajUtil.get_loss(self.shadow_test_dataset, self.shadow_model_access,
                                                         self.auxiliary_info).reshape(-1, 1)
            shadow_test_loss_trajectory = np.concatenate([original_shadow_traj, shadow_test_loss_trajectory], axis=1)
            np.save(self.auxiliary_info.shadow_losstraj_path + '/shadow_test_loss_traj.npy',
                    np.array(shadow_test_loss_trajectory))
        else:
            shadow_test_loss_trajectory = np.load(
                self.auxiliary_info.shadow_losstraj_path + '/shadow_test_loss_traj.npy', allow_pickle=True)

        print("PREPARE: Training attack model...")
        self.attack_model = attack_model(self.auxiliary_info.distillation_epochs + 1)
        self.attack_model = LosstrajUtil.train_attack_model(shadow_train_loss_trajectory, shadow_test_loss_trajectory,
                                                            self.auxiliary_info, self.attack_model, self.num_classes)

        self.prepared = True

    def infer(self, dataset) -> np.ndarray:
        """
        Infer the membership of the dataset.
        :param dataset: the dataset to infer.
        :return: the inferred membership of shape
        """
        if not self.prepared:
            raise ValueError("The attack is not prepared yet!")

        set_seed(self.auxiliary_info.seed)

        # obtain the loss trajectory of the target model
        print("INFER: Obtaining loss trajectory of the target model...")
        target_loss_trajectory = LosstrajUtil.get_loss_trajectory(dataset,
                                                                  self.target_model_access.get_untrained_model(),
                                                                  self.auxiliary_info, model_type="target")

        original_shadow_traj = LosstrajUtil.get_loss(dataset, self.shadow_model_access, self.auxiliary_info).reshape(-1,
                                                                                                                     1)
        target_loss_trajectory = np.concatenate([target_loss_trajectory, original_shadow_traj], axis=1)

        # infer the membership
        data_to_infer = np.array(target_loss_trajectory)
        data_to_infer = torch.tensor(data_to_infer, dtype=torch.float32)
        target_pred = self.attack_model(data_to_infer.to(self.auxiliary_info.device))

        target_pred = target_pred.detach().cpu().numpy()
        target_pred = np.transpose(target_pred)[0]
        return target_pred
