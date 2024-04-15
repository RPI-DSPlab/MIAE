import copy

import numpy as np
import torch
from tqdm import tqdm

from miae.attacks.attack_classifier import *


class ModelAccessType(Enum):
    """ Enum class for model access type. """
    WHITE_BOX = "white_box"
    BLACK_BOX = "black_box"
    GRAY_BOX = "gray_box"
    LABEL_ONLY = "label_only"


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

    def build_attack_classifier(self, classifer_config: dict) -> AttackClassifier:
        """
        Build the attack model. This function is called after the prepare method. It uses the signals from the
        auxiliary information to build the attack model.
        :param config: the configuration/hyperparameters of the attack model. It is a dictionary containing the necessary
        information for building the classifier including the type of the classifier, the hyperparameters of the classifier,
        parameter grid for grid search, etc. see attack_classifier.py for more details.
        :return: the attack model.
        """
        attack_classifier_type = classifer_config.get('attack_classifier_type')
        if attack_classifier_type is None:
            raise ValueError("Attack classifier type is not specified.")
        if attack_classifier_type == AttackType.LOGISTIC_REGRESSION.value:
            attack_classifier = LrAC(classifer_config)
        elif attack_classifier_type == AttackType.MULTI_LAYERED_PERCEPTRON.value:
            attack_classifier = MlpAC(classifer_config)
        elif attack_classifier_type == AttackType.RANDOM_FOREST.value:
            attack_classifier = RandomForestAC(classifer_config)
        attack_classifier.build_classifier(self.aux_sample_signals, self.aux_member_labels, self.aux_sample_weights)
        self.attack_classifier = attack_classifier
        return attack_classifier

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

                print(
                    f"Epoch {epoch}, train_acc: {train_accuracy * 100:.2f}%, test_acc: {test_accuracy * 100:.2f}%, Loss: "
                    f"{train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")

                if aux_info.log_path is not None:
                    aux_info.logger.info(
                        f"Epoch {epoch}, train_acc: {train_accuracy * 100:.2f}%, test_acc: {test_accuracy * 100:.2f}%, Loss:"
                        f"{train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
        return shadow_model

    @classmethod
    def train_attack_model(cls, attack_model, attack_train_loader, attack_test_loader, aux_info: AuxiliaryInfo) -> torch.nn.Module:
        """
        Train the attack model.
        :param attack_model: the attack model.
        :param attack_train_loader: the attack training data loader.
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
                        aux_info.logger.info(
                            f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, test_acc: {test_acc * 100:.2f}%, Loss: {train_loss:.4f}")
                else:
                    print(f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, Loss: {train_loss * 100:.4f}%")
                    if aux_info.log_path is not None:
                        aux_info.logger.info(
                            f"Epoch: {epoch}, train_acc: {train_acc * 100:.2f}%, Loss: {train_loss * 100:.4f}%")

        return attack_model

