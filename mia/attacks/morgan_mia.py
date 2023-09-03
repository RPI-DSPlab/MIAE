# This code implements "Revisiting Membership Inference Under Realistic Assumptions", PETs 2021
# The code is based on the code from
# https://github.com/bargavj/EvaluatingDPML
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from mia.utils import datasets
from mia.utils import models
from mia.utils.set_seed import set_seed


class MerlinAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of auxiliary information for Merlin attack.
    """

    def __init__(self, config):
        """
        Initialize auxiliary information with configuration.
        :param config: a dictionary containing the configuration for auxiliary information.
        """
        super().__init__(config)

        # ---- initialize auxiliary information with default values ----
        # -- Morgan attack parameters --
        self.max_t = 100  # maximum number of iterations for obtaining the merlin ratio
        self.attack_noise_type = 'gaussian'
        self.attack_noise_coverage = 'full'
        self.attack_noise_magnitude = 0.01
        self.attack_fpr_threshold = 0.05

        # -- target model parameters --

        # -- update auxiliary information with configuration --
        for key, value in config.items():
            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                raise ValueError(f"Unknown configuration: {key}")


class MerlinModelAccess(ModelAccess):
    """
    Implementation of model access for Morgan attack.
    """

    def __init__(self, model, access_type: ModelAccessType):
        """
        Initialize model access with model and access type.
        :param model: the target model.
        :param access_type: the type of access to the target model.
        """
        super().__init__(model, access_type)


class MerlinUtil:
    # To avoid numerical inconsistency in calculating log_loss
    SMALL_VALUE = 1e-6

    @classmethod
    def generate_noise(cls, shape, dtype, noise_params):
        """
        Generate noise with given shape, data type and noise parameters.
        :param dtype: data type of the noise.
        :param noise_params: a tuple of noise parameters, including noise type, noise coverage and noise magnitude.
        :return: noise with given shape, data type and noise parameters.
        """
        noise_type, noise_coverage, noise_magnitude = noise_params
        if noise_coverage == 'full':
            if noise_type == 'uniform':
                return np.array(np.random.uniform(0, noise_magnitude, size=shape), dtype=dtype)
            else:
                return np.array(np.random.normal(0, noise_magnitude, size=shape), dtype=dtype)
        attr = np.random.randint(shape[1])
        noise = np.zeros(shape, dtype=dtype)
        if noise_type == 'uniform':
            noise[:, attr] = np.array(np.random.uniform(0, noise_magnitude, size=shape[0]), dtype=dtype)
        else:
            noise[:, attr] = np.array(np.random.normal(0, noise_magnitude, size=shape[0]), dtype=dtype)
        return noise

    @classmethod
    def train_shadow_model(cls, info: MerlinAuxiliaryInfo, dataset: datasets.AbstractGeneralDataset,
                           model: models.BaseModel):
        """
        Train a shadow model. This is used to prepare the Decision Threshold for the Merlin attack.
        if the model is found in the shadow_save_dir, then load the model from the save_dir.

        :param info: auxiliary information for the attack.
        :param access: model access for the attack.
        :param dataset: the dataset for training the shadow model. It should be a subset of the target dataset distribution
        :param model: the initialized shadow model.

        :return: the shadow model.
        """

    @classmethod
    def train_target_model(cls):

    @classmethod
    def log_loss(cls, a, b):
        """
        Compute the log loss between two distributions.
        :param a: the first distribution.
        :param b: the second distribution.
        :return: the log loss between two distributions.
        """
        return [-np.log(max(b[i, int(a[i])], cls.SMALL_VALUE)) for i in range(len(a))]

    @classmethod
    def get_merlin_ratio(cls, true_x, true_y, classifier, per_instance_loss, noise_params, max_t):
        """
        Returns the merlin-ratio for the Merlin attack, the merlin-ratio
        is between 0 and 1.

        Quote from the paper:
        The intuition here is that due to overfitting, the target model’s loss on a training set record will tend to be
        close to a local minimum, so the loss at perturbed points near the original input will be higher. On the other
        hand, the loss is equally likely to either increase or decrease for a non-member record.

        :param true_x: the true input (batched).
        :param true_y: the true output (batched).
        :param classifier: the target model.
        :param per_instance_loss: the per-instance loss of the target model (loss).
        :param noise_params: a tuple of noise parameters, including noise type, noise coverage and noise magnitude.
                            This is passed to MorganUtil.generate_noise.
        :param max_t: maximum number of iterations.
        """
        counts = np.zeros(len(true_x))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for t in range(max_t):
            noisy_x = true_x + torch.tensor(cls.generate_noise(true_x.shape, true_x.dtype, noise_params), device=device)
            noisy_x = noisy_x.to(torch.float32)  # Ensure the data type is float32

            # Convert numpy arrays to PyTorch tensors
            noisy_x_tensor = torch.tensor(noisy_x, device=device)
            true_y_tensor = torch.tensor(true_y, device=device)

            # Create a TensorDataset from the tensors with the added noise
            dataset = TensorDataset(noisy_x_tensor, true_y_tensor)
            data_loader = DataLoader(dataset, batch_size=len(true_x))

            # Store predictions for all instances
            pred_y = []

            with torch.no_grad():
                for inputs, _ in data_loader:
                    inputs = inputs.unsqueeze(1)  # Add a dummy dimension for the model
                    predictions = classifier(inputs)
                    pred_y.extend(predictions.squeeze().cpu().numpy())

            pred_y = np.array(pred_y)
            noisy_per_instance_loss = np.array(cls.log_loss(true_y, pred_y))
            counts += np.where(noisy_per_instance_loss > per_instance_loss, 1, 0)

        return counts / max_t

    @classmethod
    def get_inference_threshold(cls, pred_vector, true_vector, fpr_threshold=None):
        """
        Returns the inference threshold for the Merlin attack.
        :param pred_vector: the predicted output (batched).
        :param true_vector: the true output (batched).
        :param fpr_threshold: fpr_threshold for the Merlin attack.
        :return: a threshold for the Merlin attack.
        """
        fpr, tpr, thresholds = roc_curve(true_vector, pred_vector, pos_label=1)
        # return inference threshold corresponding to maximum advantage
        if fpr_threshold == None:
            return thresholds[np.argmax(tpr - fpr)]
        # return inference threshold corresponding to fpr_threshold
        for a, b in zip(fpr, thresholds):
            if a > fpr_threshold:
                break
            alpha_thresh = b
        return alpha_thresh

    @classmethod
    def merlin_mia(cls, true_x, true_y, classifier, per_instance_loss, noise_params, max_t, fpr_threshold=None):
        """
        Implementation of the Merlin attack.
        :param true_x: the true input (batched).
        :param true_y: the true output (batched).
        :param classifier: the target model.
        :param per_instance_loss: the per-instance loss of the target model (loss).
        :param noise_params: a tuple of noise parameters, including noise type, noise coverage and noise magnitude.
                            This is passed to MorganUtil.generate_noise.
        :param max_t: maximum number of iterations.
        :param fpr_threshold: fpr_threshold for the Merlin attack.
        """
        # obtain the merlin ratio
        merlin_ratio = cls.get_merlin_ratio(true_x, true_y, classifier, per_instance_loss, noise_params, max_t)
        # obtain the inference threshold
        threshold = cls.get_inference_threshold(merlin_ratio, true_y, fpr_threshold)
        # obtain the predicted output
        pred_y = np.where(merlin_ratio > threshold, 1, 0)
        return pred_y


class MerlinAttack(MiAttack):
    """
    Implementation of the Merlin attack.
    """

    def __init__(self, target_model_access: MerlinModelAccess, auxiliary_info: MerlinAuxiliaryInfo, target_data=None):
        """
        Initialize the Morgan attack with target model access and auxiliary information.
        :param target_model_access: the target model access.
        :param auxiliary_info: the auxiliary information.
        """
        super().__init__(target_model_access, auxiliary_info, target_data)
        self.auxiliary_info = auxiliary_info
        self.target_model_access = target_model_access
        self.noise_params = (self.auxiliary_info.attack_noise_type,
                             self.auxiliary_info.attack_noise_coverage,
                             self.auxiliary_info.attack_noise_magnitude)

    def prepare(self, attack_config: dict):
        """
        Prepare the Merlin attack. This function is called before the attack. It may use model access to get signals
        from the target model.
        :param attack_config: a dictionary containing the configuration for the attack.
        :return: None.
        """

    def infer(self, target_data):
        """
        Infers the membership of target data with the Merlin attack.
        :param target_data: the target data.
        :return: the inferred membership.
        """

        # obtain the per-instance loss of the target data
