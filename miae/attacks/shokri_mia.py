# This code implements "Membership Inference Attacks against Machine Learning Models" by Shokri et al.
# https://arxiv.org/abs/1610.05820
import copy
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from sklearn.metrics import roc_curve
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from miae.utils.set_seed import set_seed


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


class ShokriAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of the auxiliary information for the Shokri attack.
    """

    def __init__(self, config, attack_model=AttackMLP):
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
        self.num_shadow_models = config.get("num_shadow_models", 4)
        self.num_shadow_epochs = config.get("num_shadow_epochs", self.epochs)
        self.shadow_batch_size = config.get("shadow_batch_size", self.batch_size)
        self.shadow_lr = config.get("shadow_lr", self.batch_size)
        self.shadow_train_ratio = config.get("shadow_train_ratio", 0.5)  # 0.5 for a balanced prior for membership

        # -- attack model parameters --
        self.attack_model = attack_model
        self.num_attack_epochs = config.get("num_attack_epochs", self.epochs)
        self.attack_batch_size = config.get("attack_batch_size", self.batch_size)
        self.attack_lr = config.get("attack_lr", 0.01)
        self.attack_train_ratio = config.get("attack_train_ratio", 0.9)
        self.attack_epochs = config.get("attack_epochs", self.epochs)

        # -- other parameters --
        self.save_path = config.get("save_path", "shokri")
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.shadow_model_path = config.get("shadow_model_path", f"{self.save_path}/shadow_models")
        self.attack_dataset_path = config.get("attack_dataset_path", f"{self.save_path}/attack_dataset")
        self.attack_model_path = config.get("attack_model_path", f"{self.save_path}/attack_models")
        self.cos_scheduler = config.get("cos_scheduler", True)  # use cosine annealing scheduler for shadow model

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)


class ShokriModelAccess(ModelAccess):
    """
    Implementation of model access for Shokri attack.
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType = ModelAccessType.BLACK_BOX):
        """
        Initialize model access with model and access type.
        :param model: the target model.
        :param access_type: the type of access to the target model.
        """
        super().__init__(model, untrained_model, access_type)


class ShokriUtil:
    @classmethod
    def train_shadow_model(cls, shadow_model, shadow_train_loader, shadow_test_loader, aux_info: ShokriAuxiliaryInfo):
        """
        Train the shadow model.
        :param shadow_model: the shadow model.
        :param shadow_train_loader: the shadow training data loader.
        :param shadow_test_loader: the shadow test data loader.
        :param aux_info: the auxiliary information for the shadow model.
        :return: the trained shadow model.
        """
        shadow_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, shadow_model.parameters()), lr=aux_info.lr,
                                           momentum=aux_info.momentum,
                                           weight_decay=aux_info.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(shadow_optimizer, aux_info.num_shadow_epochs)
        shadow_criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(aux_info.num_shadow_epochs)):
            shadow_model.train()
            train_loss = 0
            for data, target in shadow_train_loader:
                data, target = data.to(aux_info.device), target.to(aux_info.device)
                shadow_optimizer.zero_grad()
                output = shadow_model(data)
                loss = shadow_criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                shadow_optimizer.step()
            scheduler.step()

            if epoch % 20 == 0 or epoch == aux_info.num_shadow_epochs - 1:
                shadow_model.eval()
                with torch.no_grad():
                    test_correct_predictions = 0
                    total_samples = 0
                    for i, data in enumerate(shadow_test_loader):
                        inputs, labels = data
                        inputs, labels = inputs.to(aux_info.device), labels.to(aux_info.device)
                        outputs = shadow_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        test_correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    test_accuracy = test_correct_predictions / total_samples

                    train_correct_predictions = 0
                    total_samples = 0
                    for i, data in enumerate(shadow_train_loader):
                        inputs, labels = data
                        inputs, labels = inputs.to(aux_info.device), labels.to(aux_info.device)
                        outputs = shadow_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        train_correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                    train_accuracy = train_correct_predictions / total_samples

                print(
                    f"Epoch {epoch}, train_acc: {train_accuracy * 100:.2f}%, test_acc: {test_accuracy * 100:.2f}%, Loss: "
                    f"{train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")

                if aux_info.log_path is not None:
                    logging.info(
                        f"Epoch {epoch}, train_acc: {train_accuracy * 100:.2f}%, test_acc: {test_accuracy * 100:.2f}%, Loss:"
                        f"{train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
        return shadow_model

    @classmethod
    def train_attack_model(cls, attack_model, attack_train_loader, attack_test_loader, aux_info: ShokriAuxiliaryInfo):
        """
        Train the attack model.
        :param attack_model: the attack model.
        :param attack_train_loader: the attack training data loader.
        :param attack_test_loader: the attack test data loader, None meaning no test data.
        :param aux_info: the auxiliary information for the attack model.
        :return: the trained attack model.
        """
        attack_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, attack_model.parameters()),
                                           lr=aux_info.attack_lr,
                                           momentum=aux_info.momentum,
                                           weight_decay=aux_info.weight_decay)
        attack_criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(aux_info.attack_epochs)):
            attack_model.train()
            train_loss = 0
            for pred, label, membership in attack_train_loader:
                pred, membership = pred.to(aux_info.device), membership.to(aux_info.device)
                attack_optimizer.zero_grad()
                output = attack_model(pred)
                membership = membership.long()
                loss = attack_criterion(output, membership)  # membership is the target to be predicted
                loss.backward()
                attack_optimizer.step()
                train_loss += loss.item()

            if epoch % 20 == 0 or epoch == aux_info.attack_epochs - 1:
                attack_model.eval()
                correct = 0
                total = 0
                if attack_test_loader != None:
                    with torch.no_grad():
                        for pred, _, membership in attack_test_loader:
                            pred, membership = pred.to(aux_info.device), membership.to(aux_info.device)
                            output = attack_model(pred)
                            _, predicted = torch.max(output.data, 1)
                            total += membership.size(0)
                            correct += (predicted == membership).sum().item()
                    test_acc = correct / total

                with torch.no_grad():
                    correct = 0
                    total = 0
                    for pred, _, membership in attack_train_loader:
                        pred, membership = pred.to(aux_info.device), membership.to(aux_info.device)
                        output = attack_model(pred)
                        _, predicted = torch.max(output.data, 1)
                        total += membership.size(0)
                        correct += (predicted == membership).sum().item()
                    train_acc = correct / total

                if attack_test_loader != None:
                    print(
                        f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, test_acc: {test_acc * 100:.2f}%, Loss: {train_loss:.4f}")
                    if aux_info.log_path is not None:
                        logging.info(
                            f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, test_acc: {test_acc * 100:.2f}%, Loss: {train_loss:.4f}")
                else:
                    print(f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, Loss: {train_loss * 100:.4f}%")
                    if aux_info.log_path is not None:
                        logging.info(
                            f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, Loss: {train_loss * 100:.4f}%")

        return attack_model

    @classmethod
    def split_dataset(cls, dataset: Dataset, num_datasets: int) -> list[Dataset]:
        """
        Split the dataset into multiple datasets.
        :param dataset: the dataset to be split.
        :param num_datasets: the number of datasets to be split into.
        :return: a list of datasets.
        """
        # Calculate the sizes of subsets
        subset_sizes = [len(dataset) // num_datasets] * num_datasets
        for i in range(len(dataset) % num_datasets):
            subset_sizes[i] += 1

        subsets = torch.utils.data.random_split(dataset, subset_sizes)

        # Return a list of datasets
        return subsets

    @classmethod
    def filter_dataset(cls, dataset: Dataset, label: int) -> Dataset:
        """
        Filter the dataset with the specified label.
        :param dataset: the dataset to be filtered.
        :param label: the label to be filtered.
        :return: the filtered dataset.
        """

        filtered_indices = [i for i in range(len(dataset)) if dataset[i][1] == label]
        filtered_dataset = Subset(dataset, filtered_indices)
        return filtered_dataset


class AttackTrainingSet(Dataset):
    def __init__(self, predictions, class_labels, in_out):
        self.predictions = predictions  # Prediction values
        self.class_labels = class_labels  # Class labels
        self.in_out = in_out  # "in" or "out" indicator

        # ensure self.in_out is binary
        assert len(np.unique(self.in_out)) == 2, "in_out should be binary"

        # Ensure all inputs have the same length
        assert len(predictions) == len(class_labels) == len(in_out), "Lengths of inputs should match"

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        prediction = self.predictions[idx]
        class_label = self.class_labels[idx]
        in_out_indicator = self.in_out[idx]

        return prediction, class_label, in_out_indicator


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
        self.attack_model_dict = None
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

    def prepare(self, auxiliary_dataset):
        """
        Prepare the attack.
        :param auxiliary_dataset: the auxiliary dataset (will be split into training sets )
        """
        super().prepare(auxiliary_dataset)
        if self.prepared:
            print("The attack has already prepared!")
            return

        self.attack_model = self.auxiliary_info.attack_model(self.auxiliary_info.num_classes)
        self.attack_model_dict = {}

        # set seed
        set_seed(self.auxiliary_info.seed)

        if self.auxiliary_info.log_path is not None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filename=f'{self.auxiliary_info.log_path}/shokri_mia.log',
            )

        # create shadow datasets
        sub_shadow_dataset_list = ShokriUtil.split_dataset(auxiliary_dataset, self.auxiliary_info.num_shadow_models)

        # step 1: train shadow models
        if not os.path.exists(self.auxiliary_info.attack_dataset_path):
            # if attack dataset exists, then there's no need to retrain shadow models
            in_prediction_set_pred = None
            in_prediction_set_label = None
            out_prediction_set_pred = None
            out_prediction_set_label = None

            for i in range(self.auxiliary_info.num_shadow_models):
                # train k shadows models to build attack dataset
                model_name = f"shadow_model_{i}.pt"
                model_path = os.path.join(self.auxiliary_info.shadow_model_path, model_name)

                shadow_model_i = self.target_model_access.get_untrained_model()
                shadow_model_i.to(self.auxiliary_info.device)
                train_len = int(len(sub_shadow_dataset_list[i]) * self.auxiliary_info.shadow_train_ratio)
                test_len = len(sub_shadow_dataset_list[i]) - train_len
                shadow_train_dataset, shadow_test_dataset = torch.utils.data.random_split(sub_shadow_dataset_list[i],
                                                                                          [train_len, test_len])

                shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=self.auxiliary_info.shadow_batch_size,
                                                 shuffle=True)
                shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=self.auxiliary_info.shadow_batch_size,
                                                shuffle=False)
                if os.path.exists(model_path):
                    print(f"Loading shadow model {i + 1}/{self.auxiliary_info.num_shadow_models}...")
                    if self.auxiliary_info.log_path is not None:
                        logging.info(f"Loading shadow model {i + 1}/{self.auxiliary_info.num_shadow_models}...")
                    shadow_model_i.load_state_dict(torch.load(model_path))
                else:
                    print(f"Training shadow model {i + 1}/{self.auxiliary_info.num_shadow_models}...")
                    if self.auxiliary_info.log_path is not None:
                        logging.info(f"Training shadow model {i + 1}/{self.auxiliary_info.num_shadow_models}...")
                    shadow_model_i = ShokriUtil.train_shadow_model(shadow_model_i, shadow_train_loader,
                                                                   shadow_test_loader,
                                                                   self.auxiliary_info)
                    torch.save(shadow_model_i.state_dict(), model_path)

                # building the attack dataset
                for data, target in shadow_train_loader:
                    data, target = data.to(self.auxiliary_info.device), target.to(self.auxiliary_info.device)
                    output = shadow_model_i(data)
                    if in_prediction_set_pred is None:  # first entry
                        in_prediction_set_pred = output.cpu().detach().numpy()
                        in_prediction_set_label = target.cpu().detach().numpy()
                    else:
                        in_prediction_set_pred = np.concatenate((in_prediction_set_pred, output.cpu().detach().numpy()))
                        in_prediction_set_label = np.concatenate(
                            (in_prediction_set_label, target.cpu().detach().numpy()))

                for data, target in shadow_test_loader:
                    data, target = data.to(self.auxiliary_info.device), target.to(self.auxiliary_info.device)
                    output = shadow_model_i(data)
                    if out_prediction_set_pred is None:  # first entry
                        out_prediction_set_pred = output.cpu().detach().numpy()
                        out_prediction_set_label = target.cpu().detach().numpy()
                    else:
                        out_prediction_set_pred = np.concatenate(
                            (out_prediction_set_pred, output.cpu().detach().numpy()))
                        out_prediction_set_label = np.concatenate(
                            (out_prediction_set_label, target.cpu().detach().numpy()))

            # step 2: create attack dataset for attack model training
            in_prediction_set_membership = np.ones(len(in_prediction_set_pred))
            out_prediction_set_membership = np.zeros(len(out_prediction_set_pred))

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

        if self.auxiliary_info.attack_train_ratio < 1.0:
            attack_train_dataset, attack_test_dataset = torch.utils.data.random_split(self.attack_dataset,
                                                                                      [train_len, test_len])
        else:
            attack_train_dataset = self.attack_dataset
            attack_test_dataset = None

        # train attack models for each label
        labels = np.unique(self.attack_dataset.class_labels)
        # if attack model exists, then there's no need to retrain attack models
        if len(labels) == len(os.listdir(self.auxiliary_info.attack_model_path)):
            print("Loading attack models...")
            if self.auxiliary_info.log_path is not None:
                logging.info("Loading attack models...")
            for i, label in enumerate(labels):
                model = self.auxiliary_info.attack_model(self.auxiliary_info.num_classes)
                model.load_state_dict(torch.load(f"{self.auxiliary_info.attack_model_path}/attack_model_{label}.pt"))
                model.to(self.auxiliary_info.device)
                self.attack_model_dict[label] = model
        else:
            for i, label in enumerate(labels):
                print(f"Training attack model for {i}/{len(labels)} label \"{label}\" ...")
                if self.auxiliary_info.log_path is not None:
                    logging.info(f"Training attack model for {i}/{len(labels)} label \"{label}\" ...")
                # filter the dataset with the label
                attack_train_dataset_filtered = ShokriUtil.filter_dataset(attack_train_dataset, label)
                attack_test_dataset_filtered = ShokriUtil.filter_dataset(attack_test_dataset,
                                                                         label) if attack_test_dataset else None
                self.attack_train_loader = DataLoader(attack_train_dataset_filtered,
                                                      batch_size=self.auxiliary_info.attack_batch_size,
                                                      shuffle=True)
                self.attack_test_loader = DataLoader(attack_test_dataset_filtered,
                                                     batch_size=self.auxiliary_info.attack_batch_size,
                                                     shuffle=True) if attack_test_dataset else None
                untrained_attack_model = self.auxiliary_info.attack_model(self.auxiliary_info.num_classes)
                untrained_attack_model.to(self.auxiliary_info.device)
                trained_attack_model = ShokriUtil.train_attack_model(untrained_attack_model, self.attack_train_loader,
                                                                     self.attack_test_loader,
                                                                     self.auxiliary_info)
                self.attack_model_dict[label] = trained_attack_model
                torch.save(trained_attack_model.state_dict(),
                           f"{self.auxiliary_info.attack_model_path}/attack_model_{label}.pt")

        self.prepared = True

    def infer(self, target_data) -> np.ndarray:
        """
        Infer the membership of the target data.
        """
        super().infer(target_data)
        if not self.prepared:
            raise ValueError("The attack has not been prepared!")

        # load the attack models
        labels = np.unique(self.attack_dataset.class_labels)
        for label in labels:
            if label not in self.attack_model_dict:
                model = self.auxiliary_info.attack_model(self.auxiliary_info.num_classes)
                model.load_state_dict(torch.load(f"{self.auxiliary_info.attack_model_path}/attack_model_{label}.pt"))
                model.to(self.auxiliary_info.device)
                self.attack_model_dict[label] = model

        # infer the membership
        self.target_model_access.model.to(self.auxiliary_info.device)
        membership = []

        target_data_loader = DataLoader(target_data, batch_size=self.auxiliary_info.batch_size, shuffle=False)

        for data, target in target_data_loader:
            data = data.to(self.auxiliary_info.device)
            output = self.target_model_access.model(data)
            output = output.cpu().detach().numpy()
            for i, label in enumerate(target):
                label = label.item()
                member_pred = self.attack_model_dict[label](torch.tensor(output[i]).to(self.auxiliary_info.device))
                member_pred = member_pred.cpu().detach().numpy()
                membership.append(member_pred.reshape(-1))

        return np.array(np.transpose(membership)[0])
