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
from typing import List, Optional, Union

from mia.utils.set_seed import set_seed
from mia.utils.dataset_utils import get_num_classes, dataset_split
from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack


class AttackMLP(torch.nn.Module):
    # default model for the attack
    def __init__(self, dim_in):
        super(AttackMLP, self).__init__()
        self.dim_in = dim_in
        self.fc1 = torch.nn.Linear(self.dim_in, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, 2)

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
        self.batch_size = config.get('batch_size', 500)
        self.num_workers = config.get('num_workers', 2)
        self.distillation_epochs = config.get('distillation_epochs', 200)

        # directories:
        self.save_path = config.get('save_path', './losstraj_files')
        self.distill_models_path = self.save_path + '/distill_models'
        self.shadow_model_path = self.save_path + '/shadow_model.pth'
        self.shadow_losstraj_path = self.save_path + '/shadow_losstraj'

        # dataset length: it should be given as the ratio of the training dataset length w.r.t. the whole auxiliary dataset
        self.distillation_dataset_ratio = config.get('distillation_dataset_ratio', 0.5)
        self.shadow_dataset_ratio = 1 - self.distillation_dataset_ratio
        # train/test split ratio of the shadow dataset
        self.shadow_train_test_split_ratio = config.get('shadow_train_test_split_ratio', 0.5)
        self.num_classes = config.get('num_classes', 10)

        self.attack_model = attack_model


class LosstrajModelAccess(ModelAccess):
    """
    The model access for the loss trajectory based membership inference attack.
    """

    def __init__(self, model, model_type: ModelAccessType = ModelAccessType.BLACK_BOX):
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
        model_copy = copy.deepcopy(self.model)
        for layer in model_copy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        return model_copy


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
        if os.path.exists(os.path.join(auxiliary_info.distill_models_path, teacher_type)) == False:
            os.makedirs(os.path.join(auxiliary_info.distill_models_path, teacher_type))
        elif len(os.listdir(
                os.path.join(auxiliary_info.distill_models_path, teacher_type))) >= auxiliary_info.distillation_epochs:
            return  # if the distilled models are already saved, return

        distilled_model = teacher_model_access.get_model_architecture()
        distilled_model.to(auxiliary_info.device)
        teacher_model_access.to_device(auxiliary_info.device)
        distilled_model.train()
        optimizer = torch.optim.Adam(distilled_model.parameters(), lr=0.001)

        distill_train_loader = DataLoader(distillation_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                          num_workers=auxiliary_info.num_workers)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            for i, data in enumerate(distill_train_loader):
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                teacher_pred = teacher_model_access.model(inputs)  # teacher model
                distilled_pred = distilled_model(inputs)  # student model
                loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(distilled_pred, dim=1),
                                                                 F.softmax(teacher_pred, dim=1))

                loss.backward()
                optimizer.step()

            # Calculate the accuracy for this batch
            distilled_model.eval()
            correct_predictions = 0
            total_samples = 0
            if epoch % 20 == 0 or epoch == auxiliary_info.distillation_epochs - 1:
                with torch.no_grad():
                    for i, data in enumerate(distill_train_loader):
                        inputs, labels = data
                        inputs = inputs.to(auxiliary_info.device)
                        labels = labels.to(auxiliary_info.device)
                        outputs = distilled_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    accuracy = correct_predictions / total_samples
                    print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%, Loss: {loss.item():.4f}")

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
        if os.path.exists(auxiliary_info.shadow_model_path):
            shadow_model.load_state_dict(torch.load(auxiliary_info.shadow_model_path))
            return LosstrajModelAccess(shadow_model, ModelAccessType.BLACK_BOX)

        shadow_model.to(auxiliary_info.device)
        shadow_model.train()
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                         num_workers=auxiliary_info.num_workers)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=auxiliary_info.batch_size, shuffle=True,
                                        num_workers=auxiliary_info.num_workers)

        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            for i, data in enumerate(shadow_train_loader):
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(auxiliary_info.device), labels.to(auxiliary_info.device)
                outputs = shadow_model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

            # Step the learning rate scheduler
            scheduler.step()

            # Calculate the accuracy for this batch
            if epoch % 20 == 0 or epoch == auxiliary_info.distillation_epochs - 1:
                shadow_model.eval()
                correct_predictions = 0
                total_samples = 0
                with torch.no_grad():
                    for i, data in enumerate(shadow_test_loader):
                        inputs, labels = data
                        inputs, labels = inputs.to(auxiliary_info.device), labels.to(auxiliary_info.device)
                        outputs = shadow_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    accuracy = correct_predictions / total_samples
                    print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%, Loss: {loss.item():.4f}")

        # save the model
        torch.save(shadow_model.state_dict(), auxiliary_info.shadow_model_path)
        return LosstrajModelAccess(shadow_model, ModelAccessType.BLACK_BOX)

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

        if model_type not in ["target", "shadow"]:
            raise ValueError("model_type should be either 'target' or 'shadow'!")

        # create loader for the dataset
        data_loader = DataLoader(data, batch_size=auxiliary_info.batch_size, shuffle=False,
                                 num_workers=auxiliary_info.num_workers)

        # load each distilled model and record the loss trajectory
        loss_trajectory = []
        for epoch in tqdm(range(auxiliary_info.distillation_epochs)):
            distilled_model = model
            distilled_model.to(auxiliary_info.device)
            distilled_model.load_state_dict(
                torch.load(os.path.join(auxiliary_info.distill_models_path, model_type,
                                        "distilled_model_ep" + str(epoch) + ".pth")))
            distilled_model.eval()
            with torch.no_grad():
                loss_array = []
                for i, data in enumerate(data_loader):
                    inputs, labels = data
                    inputs = inputs.to(auxiliary_info.device)
                    labels = labels.to(auxiliary_info.device)
                    outputs = distilled_model(inputs)
                    loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                    loss_array.append(loss.cpu().numpy())
                loss_array = np.concatenate(loss_array, axis=0)
                loss_trajectory.append(loss_array.flatten())
        return np.array(loss_trajectory).transpose()

    @classmethod
    def get_loss(cls, data, model_access: LosstrajModelAccess, auxiliary_info: LosstrajAuxiliaryInfo):
        """
        Get the loss of model on given data.
        :param data: the dataset to obtain the loss.
        :param model_access: the model access.
        :param auxiliary_info: the auxiliary information.
        :return:
        """

        # create loader for the dataset
        data_loader = DataLoader(data, batch_size=auxiliary_info.batch_size, shuffle=False,
                                 num_workers=auxiliary_info.num_workers)

        model_access.to_device(auxiliary_info.device)
        model_access.model.eval()
        with torch.no_grad():
            loss_array = []
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs = inputs.to(auxiliary_info.device)
                labels = labels.to(auxiliary_info.device)
                outputs = model_access.model(inputs)
                loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                loss_array.append(loss.cpu().numpy())
            loss_array = np.concatenate(loss_array, axis=0)
        return np.array(loss_array)

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
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

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
                    print(f"Epoch {epoch + 1}, Accuracy(on training set): {accuracy:.2f}%, Loss: {loss.item():.4f}")
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
        self.shadow_model = self.target_model_access.get_model_architecture()
        self.shadow_model_access = LosstrajUtil.train_shadow_model(self.shadow_model, self.shadow_train_dataset,
                                                                   self.shadow_test_dataset,
                                                                   self.auxiliary_info)
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
            shadow_train_loss_trajectory = np.concatenate([shadow_train_loss_trajectory, original_shadow_traj], axis=1)
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
            shadow_test_loss_trajectory = np.concatenate([shadow_test_loss_trajectory, original_shadow_traj], axis=1)
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
                                                                  self.target_model_access.get_model_architecture(),
                                                                  self.auxiliary_info, model_type="target")

        original_shadow_traj = LosstrajUtil.get_loss(dataset, self.shadow_model_access, self.auxiliary_info).reshape(-1,
                                                                                                                     1)
        target_loss_trajectory = np.concatenate([target_loss_trajectory, original_shadow_traj], axis=1)

        # infer the membership
        data_to_infer = np.array(target_loss_trajectory)
        data_to_infer = torch.tensor(data_to_infer, dtype=torch.float32)
        target_pred = self.attack_model(data_to_infer.to(self.auxiliary_info.device))

        target_pred = target_pred.detach().cpu().numpy()
        target_pred = np.transpose(target_pred)[1]
        return target_pred
