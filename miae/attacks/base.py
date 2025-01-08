import copy

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import pickle
from enum import Enum
from abc import ABC, abstractmethod


class ModelAccessType(Enum):
    """ Enum class for model access type. """
    WHITE_BOX = "white_box"
    BLACK_BOX = "black_box"
    GRAY_BOX = "gray_box"
    LABEL_ONLY = "label_only"

class AttackTrainingSet(Dataset):
    """
    A dataset class for training the attack model. (for shokri, Boundary) It's designed to be used
    with MiAUtils.train_attack_model, as the AttackTrainingSet[1] is class label
    """
    def __init__(self, predictions, class_labels, in_out):
        if not (predictions.shape[0] == class_labels.shape[0] == in_out.shape[0]):
            raise ValueError("Lengths of inputs should match")
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

class AuxiliaryInfo(ABC):
    """
    Base class for all auxiliary information.
    """

    def __init__(self, auxiliary_info):
        """
        Initialize auxiliary information.
        :param auxiliary_info: the auxiliary information.
        """
        pass

    def save_config_to_dict(self):
        """
        Save the configuration of the auxiliary information to a dictionary.
        :return: the dictionary containing the configuration of the auxiliary information.
        """
        attr_vars = vars(self)
        attr_dict = dict()
        for key, value in vars(self).items():
            if isinstance(value, (int, float, str, bool, list, dict, np.ndarray)):
                attr_dict[key] = value

        return attr_dict


class ModelAccess(ABC):
    """
    Base class for all types of model access.
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType):
        """
        Initialize model access with a model handler.
        :param model: the model handler to be used, which can be a model object (white box) or a model api(black box).
        :param untrained_model: the untrained model handler to be used, which can be a model object (white box) or a model api(black box).
        :param type: the type of model access, which can be "white_box" or "black_box" or "gray_box"
        """
        self.model = model
        self.access_type = access_type
        self.untrained_model = untrained_model

    def get_signal(self, data):
        """
        Use model to get signal from data. The signal can be the output of a layer, or the logits,
        or the loss, or the probability vector, depending on what items attacks use.
        :param data:
        :return:
        """
        if self.access_type == ModelAccessType.BLACK_BOX:
            with torch.no_grad():
                return self.model(data)
        elif self.access_type in [ModelAccessType.WHITE_BOX, ModelAccessType.GRAY_BOX]:
            # Here, we assume that the white-box or gray-box access allows us to get
            # additional information from the model. What information we get will depend
            # on the specifics of the MIA attack and the model.
            raise NotImplementedError("White-box and gray-box access not implemented.")

        elif self.access_type == ModelAccessType.LABEL_ONLY:
            # Here, we assume that the target model only provides the label of the data.
            with torch.no_grad():
                return self.model(data).argmax(dim=1)
        else:
            raise ValueError(f"Unknown access type: {self.access_type}")

    def get_signal_lira(self, dataloader, device, augmentation='mirror'):
        """
        Generates logits for a dataloader given a model. Queries with augmentation is first
        introduced by Choquette-Choo et al. in the paper "Label-Only Membership Inference Attacks".
        Carlini et al. first brought the idea of queries with augmentation to likelihood
        ratio attack in the paper "Membership Inference Attacks From First Principles". It's
        then used for other attack such as Attack-R by Ye et al. and RMIA by Sajjad et al.

        :param dataloader: the dataloader to generate logits for.
        :param device: the device to use.
        :param augmentation: the augmentation to use. Default is 'mirror'. It could also be
        18 for the desired augmentations used for RMIA attack.

        :return: the logits for the dataloader.
        """

        def mirror_augmentation(image):
            """
            Mirrors the image.
            """
            return torch.flip(image, [2])

        def shift_augmentation(image, shift=1):
            """
            Applies shifting augmentation to the image.
            """
            padded_image = torch.nn.functional.pad(image, (shift, shift, shift, shift), mode='reflect')
            shifts = []
            for dx in range(0, 2 * shift + 1):
                for dy in range(0, 2 * shift + 1):
                    shifted = padded_image[:, :, dx:dx + 32, dy:dy + 32]
                    shifts.append(shifted)
            return shifts

        self.model.eval()
        all_logits = []


        with torch.inference_mode():
            for images, _ in dataloader:
                images = images.to(device)
                outputs = []
                if augmentation == 'none' or augmentation == 0 or augmentation == 1:
                    outputs = [self.model(images)]  # (batch_size, num_classes)

                # Apply mirror augmentation
                elif augmentation == 'mirror' or augmentation == 2:
                    mirror_images = mirror_augmentation(images)
                    output = self.model(images)
                    mirror_outputs = self.model(mirror_images)  # (batch_size, num_classes)
                    outputs = [output, mirror_outputs]

                # Apply shift augmentations
                elif augmentation == 18:
                    # Apply shift augmentations to the original image
                    shift_images = shift_augmentation(images, shift=1)
                    for shift_image in shift_images:
                        shift_outputs = self.model(shift_image)
                        outputs.append(shift_outputs)

                    # Apply shift augmentations to the mirrored image
                    mirror_images = mirror_augmentation(images)
                    shift_mirror_images = shift_augmentation(mirror_images, shift=1)
                    for shift_mirror_image in shift_mirror_images:
                        shift_mirror_outputs = self.model(shift_mirror_image)
                        outputs.append(shift_mirror_outputs)
                else:
                    raise ValueError(f"Unknown augmentation type: {augmentation}")

                # Stack all outputs along a new dimension
                all_logits.append(torch.stack(outputs, dim=1))

        # Concatenate all logits from all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_logits = all_logits.unsqueeze(1)
        return all_logits
    

    def __call__(self, data):
        return self.get_signal(data)

    def to(self, device):
        """
        Move the model to the device.
        :param device:
        :return:
        """
        self.model.to(device)

    def to_device(self, device):
        """
        Move the model to the device.
        :param device:
        :return:
        """
        self.model.to(device)

    def get_untrained_model(self):
        return copy.deepcopy(self.untrained_model)

    def eval(self):
        """
        Set the model to evaluation mode.
        :return:
        """
        self.model.eval()


class MiAttack(ABC):
    """
    Base class for all attacks.
    """

    # define initialization with specifying the model access and the auxiliary information
    def __init__(self, target_model_access: ModelAccess, auxiliary_info: AuxiliaryInfo):
        """
        Initialize the attack with model access and auxiliary information.
        :param target_model_access:
        :param auxiliary_info:
        :param target_data: if target_data is not None, the attack could be data dependent. The target data is used to
        develop the attack model or classifier.
        """
        self.target_model_access = target_model_access
        self.auxiliary_info = auxiliary_info

        self.prepared = False

    @abstractmethod
    def prepare(self, attack_config: dict):
        """
        Prepare the attack. This function is called before the attack. It may use model access to get signals
        from auxiliary information, and then uses the signals to train the attack.
        Use the auxiliary information to build shadow models/shadow data/or any auxiliary modes/information
        that are needed for building attack model or decision function.
        require set the following attributes:
        self.aux_sample_signals: the signals from the auxiliary information.
        self.aux_member_labels: the labels of the auxiliary information.
        self.aux_sample_weights: the sample weights of the auxiliary information.

        :param attack_config: the configuration/hyperparameters of the attack. It is a dictionary containing the necessary
        information. For example, the number of shadow models, the number of shadow data, etc.
        :return: everything that is needed for building attack model or decision function.
        """
        pass

    @abstractmethod
    def infer(self, target_data):
        """
        Infer the membership of data. This function is called after the prepare method. It uses the attack models or
        decision functions generated by "prepare" method to infer the membership of the target data.
        1. get signals of target data. 2. use attack_classifier to infer the membership.
        :param target_data: the data to be inferred.
        :return: the inferred membership of the data.
        """
        pass

    def save(self, path: str):
        """
        Save the attack as a pickle file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @staticmethod
    def load(path: str) -> 'MiAttack':
        """
        Load the attack from a pickle file.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

class MIAUtils:
    """
    Utils for MIA.
    Note that utils here are not for all attacks, but for some specific attacks.
    """
    @classmethod
    def log(cls, aux_info: AuxiliaryInfo, msg: str, print_flag: bool = True):
        """
        log the message to logger if the log_path is not None.
        :param aux_info: the auxiliary information.
        :param msg: the message to be logged.
        :param print_flag: whether to print the message.
        """
        if aux_info.log_path is not None:
            aux_info.logger.info(msg)
        if print_flag:
            print(msg)


    @classmethod
    def generate_keeps_lira(cls, dataset_size: int, num_experiment: int, expid:int):
        """
        This function generates the keeps for lira and all lira-inspired attacks. It generates keeps
        array to represent the index of datapoints is used for training for this experiment. This
        function is crucial to guarantees that each sample is sampled for the sames times for `num_experiment`
        times of model training. This function is adapted from lira's repo.

        :param dataset_size: the size of the dataset to generate keep for
        :param num_experiment: the number of experiments to generate keep for
        :param expid: the experiment id
        :return: the keeps array and the non-keep array
        """

        keep = np.random.uniform(0, 1, size=(num_experiment, dataset_size))
        order = keep.argsort(0)
        keep = order < int(0.5 * num_experiment)
        keep = np.array(keep[expid], dtype=bool)
        return np.where(keep), np.where(~keep)

    @classmethod
    def train_shadow_model(cls, shadow_model, shadow_train_loader, shadow_test_loader, aux_info: AuxiliaryInfo) -> torch.nn.Module:
        """
        Train the shadow model. (for shokri, Yeom, Boundary)
        :param shadow_model: the shadow model.
        :param shadow_train_loader: the shadow training data loader.
        :param shadow_test_loader: the shadow test data loader.
        :param aux_info: the auxiliary information for the shadow model.
        :return: the trained shadow model.
        """
        shadow_model.to(aux_info.device)
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

                cls.log(aux_info, f"Epoch {epoch}, train_acc: {train_accuracy * 100:.2f}%, test_acc: {test_accuracy * 100:.2f}%, Loss:"
                        f"{train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}", print_flag=True)
        return shadow_model

    @classmethod
    def train_attack_model(cls, attack_model, attack_train_loader, attack_test_loader, aux_info: AuxiliaryInfo) -> torch.nn.Module:
        """
        Train the attack model. (for shokri, Boundary) Note that loader must be AttackTrainingSet.
        :param attack_model: the attack model.
        :param attack_train_loader: the attack training data loader. It should be an Dataloader of AttackTrainingSet.
        :param attack_test_loader: the attack test data loader, None meaning no test data.
        :param aux_info: the auxiliary information for the attack model.
        :return: the trained attack model.
        """
        attack_model.to(aux_info.device)
        attack_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, attack_model.parameters()),
                                           lr=aux_info.attack_lr,
                                           momentum=aux_info.momentum,
                                           weight_decay=aux_info.weight_decay)
        attack_criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(aux_info.attack_epochs)):
            attack_model.train()
            train_loss = 0
            for pred, _, membership in attack_train_loader:
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
                    cls.log(aux_info, f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, test_acc: {test_acc * 100:.2f}%, Loss: {train_loss:.4f}", print_flag=True)
                else:
                    cls.log(aux_info, f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, Loss: {train_loss:.4f}", print_flag=True)

        return attack_model

    @classmethod
    def filter_dataset(cls, dataset: Dataset, label: int) -> Dataset:
        """
        Filter the dataset with the specified label. (for shokri, Boundary)
        :param dataset: the dataset to be filtered.
        :param label: the label to be filtered.
        :return: the filtered dataset.
        """

        filtered_indices = [i for i in range(len(dataset)) if dataset[i][1] == label]
        filtered_dataset = Subset(dataset, filtered_indices)
        return filtered_dataset

