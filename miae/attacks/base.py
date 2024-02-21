import copy

import numpy as np
import torch

from miae.attacks.attack_classifier import *


class ModelAccessType(Enum):
    """ Enum class for model access type. """
    WHITE_BOX = "white_box"
    BLACK_BOX = "black_box"
    GRAY_BOX = "gray_box"


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
    def __init__(self, target_model_access: ModelAccess, auxiliary_info: AuxiliaryInfo, target_data=None):
        """
        Initialize the attack with model access and auxiliary information.
        :param target_model_access:
        :param auxiliary_info:
        :param target_data: if target_data is not None, the attack could be data dependent. The target data is used to
        develop the attack model or classifier.
        """
        self.target_model_access = target_model_access
        self.auxiliary_info = auxiliary_info
        self.target_data = target_data

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
