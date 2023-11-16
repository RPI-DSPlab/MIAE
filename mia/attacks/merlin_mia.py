# This code implements "Revisiting Membership Inference Under Realistic Assumptions", PETs 2021
# The code is based on the code from
# https://github.com/bargavj/EvaluatingDPML
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve
from tqdm import tqdm

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from mia.utils import datasets
from mia.utils import models
from mia.utils.set_seed import set_seed
from utils.datasets import AbstractGeneralDataset


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
        self.max_t = config.get('max_t', 100)  # maximum number of iterations for obtaining the merlin ratio
        self.attack_noise_type = config.get("attack_noise_type", 'gaussian')
        self.attack_noise_coverage = config.get("attack_noise_coverage", 'full')
        self.attack_noise_magnitude = config.get("attack_noise_magnitude", 0.01)
        self.attack_fpr_threshold = config.get("attack_fpr_threshold", 0.05)

        # Model saving and loading parameters
        self.path = config.get('path', "merlin_mia_files")

        # -- shadow model parameters --
        self.shadow_model_num_epochs = config.get("shadow_model_num_epochs", 250)
        self.shadow_model_batch_size = config.get("shadow_model_batch_size", 128)
        self.shadow_model_seed = config.get("shadow_model_seed", 0)
        self.shadow_model_weight_decay = config.get('weight_decay', 0.01)
        self.shadow_model_decay = config.get('decay', 0.9999)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.shadow_path = config.get('shadow_path', f"{self.path}/shadow_model.pth")


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
    def train_shadow_model(cls, info: MerlinAuxiliaryInfo, v_dataset, model):
        """
        Train a shadow model. This is used to prepare the Decision Threshold for the Merlin attack.
        if the model is found in the shadow_save_dir, then load the model from the save_dir.

        :param info: auxiliary information for the attack.
        :param v_dataset: the dataset for training the shadow model. It should be a subset of the target dataset distribution
        :param model: the initialized shadow model.

        :return: the shadow model, the training data loader and the testing data loader.
        """

        v_classifier = copy.deepcopy(model)
        v_classifier.to(info.device)
        v_train_loader, v_test_loader = DataLoader(v_dataset[0], batch_size=info.shadow_model_batch_size, shuffle=True), \
            DataLoader(v_dataset[1], batch_size=info.shadow_model_batch_size, shuffle=True)
        # if the shadow model is available, load the shadow model from the shadow_save_dir

        if os.path.exists(info.shadow_path):
            v_classifier.load_state_dict(torch.load(info.shadow_path))
            return v_classifier, v_train_loader, v_test_loader

        # if the shadow model is not available, train the shadow model
        set_seed(info.shadow_model_seed)
        v_classifier.train()
        v_optimizer = torch.optim.Adam(v_classifier.parameters(), lr=0.001, weight_decay=info.shadow_model_weight_decay)
        v_scheduler = torch.optim.lr_scheduler.ExponentialLR(v_optimizer, gamma=info.shadow_model_decay)
        v_criterion = torch.nn.CrossEntropyLoss()

        print(f"training shadow model for {info.shadow_model_num_epochs} epochs")
        for epoch in range(info.shadow_model_num_epochs):
            for inputs, labels in v_train_loader:
                inputs, labels = inputs.to(info.device), labels.to(info.device)
                v_optimizer.zero_grad()
                outputs = v_classifier(inputs)
                loss = v_criterion(outputs, labels)
                loss.backward()
                v_optimizer.step()
            v_scheduler.step()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            correct_predictions = (predicted == labels).sum().item()
            total_samples = labels.size(0)
            accuracy = (correct_predictions / total_samples) * 100
            if epoch % 20 == 0:
                print(f"Epoch {epoch + 1},\tAccuracy: {accuracy:.2f}%,\tLoss: {loss.item():.4f}")
        torch.save(v_classifier.state_dict(), info.shadow_path)

        return v_classifier, v_train_loader, v_test_loader

    @classmethod
    def log_loss(cls, true_y, pred_y):
        """
        Compute the log loss between predictions and true labels.
        :param true_y: true label of the data.
        :param pred_y: predicted label of the data.
        :return: the log loss between predictions and true labels.
        """
        return [-np.log(max(pred_y[i, int(true_y[i])], cls.SMALL_VALUE)) for i in range(len(true_y))]

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

        print("getting merlin ratio")
        for t in tqdm(range(max_t)):
            noise = torch.tensor(cls.generate_noise(true_x.shape, true_x.dtype, noise_params), device=device)
            noisy_x = true_x + noise.cpu().numpy()
            noisy_x = noisy_x.astype(np.float32)  # Ensure the data type is float32

            # Convert numpy arrays to PyTorch tensors
            noisy_x_tensor = torch.tensor(noisy_x, device=device)
            true_y_tensor = torch.tensor(true_y, device=device)

            # Create a TensorDataset from the tensors with the added noise
            dataset = TensorDataset(noisy_x_tensor, true_y_tensor)
            data_loader = DataLoader(dataset, batch_size=128)

            # Store predictions for all instances
            pred_y = []

            with torch.no_grad():
                for inputs, _ in data_loader:
                    permuted_input = inputs.permute(0, 3, 1, 2)
                    predictions = classifier(permuted_input)
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
    def generate_logits(cls, model, data_loader, device):
        """
        Generate logits for the Merlin attack.
        :param model:
        :param data_loader: a data loader for the target data.
        :param device: the device to use.
        :return: logits for the Merlin attack.
        """
        model.eval()
        logits = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, _ = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                logits.extend(outputs.squeeze().cpu().numpy())

        return np.array(logits)

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

    def __init__(self, target_model_access: MerlinModelAccess, auxiliary_info: MerlinAuxiliaryInfo, shadow_model):
        """
        Initialize the Morgan attack with target model access and auxiliary information.
        :param target_model_access: the target model access.
        :param auxiliary_info: the auxiliary information.
        :param shadow_model: the shadow model (untrained).
        """
        super().__init__(target_model_access, auxiliary_info)
        self.v_test_loader = None
        self.v_train_loader = None
        self.merlin_ratio = None
        self.v_merlin_ratio = None
        self.auxiliary_info = auxiliary_info
        self.target_model_access = target_model_access
        self.noise_params = (self.auxiliary_info.attack_noise_type,
                             self.auxiliary_info.attack_noise_coverage,
                             self.auxiliary_info.attack_noise_magnitude)

        self.shadow_model = copy.deepcopy(shadow_model)  # copy the untrained shadow model

        self.prepared = False

    def prepare(self, v_dataset: list):
        """
        Prepare the Merlin attack. This function is called before the attack. It may use model access to get signals
        from the target model.
        :param v_dataset: the shadow dataset [train, test]
        :return: None.
        """

        # initializing the directory for saving the shadow model
        for dir in [self.auxiliary_info.path]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.shadow_model, self.v_train_loader, self.v_test_loader = MerlinUtil.train_shadow_model(self.auxiliary_info,
                                                                                                   v_dataset,
                                                                                                   self.shadow_model)
        # combine the train and test set
        v_train_set = v_dataset[0].dataset.dataset
        v_test_set = v_dataset[1].dataset.dataset
        v_dataset_data = np.concatenate([v_train_set.data, v_test_set.data], axis=0)
        v_dataset_cat = torch.utils.data.ConcatDataset([v_train_set, v_test_set])
        v_dataset_targets = v_train_set.targets + v_test_set.targets
        # go through the shadow set to get the merlin ratio
        v_pred_y = MerlinUtil.generate_logits(self.shadow_model, DataLoader(v_dataset_cat, batch_size=128),
                                              self.auxiliary_info.device)
        v_per_instance_loss = np.array(MerlinUtil.log_loss(v_dataset_targets, v_pred_y))
        noise_params = (self.auxiliary_info.attack_noise_type,
                        self.auxiliary_info.attack_noise_coverage,
                        self.auxiliary_info.attack_noise_magnitude)

        # check if the merlin ratio is already calculated
        if os.path.exists(os.path.join(self.auxiliary_info.path, "merlin_ratio.npy")):
            self.v_merlin_ratio = np.load(os.path.join(self.auxiliary_info.path, "merlin_ratio.npy"))
        else:

            self.v_merlin_ratio = MerlinUtil.get_merlin_ratio(v_dataset_data, v_dataset_targets, self.shadow_model,
                                                              v_per_instance_loss, noise_params,
                                                              self.auxiliary_info.max_t)
            np.save(os.path.join(self.auxiliary_info.path, "merlin_ratio.npy"), self.v_merlin_ratio)

        # prepare the shadow model's logits

        self.v_logits = []  # the logits of the shadow set
        self.v_y = []  # the membership of the shadow set
        # train dataset is in-sample
        train_logits = MerlinUtil.generate_logits(self.shadow_model, self.v_train_loader, self.auxiliary_info.device)
        self.v_logits.append(train_logits)
        self.v_y.append(torch.ones(train_logits.shape[0], dtype=torch.long, device=self.auxiliary_info.device))
        # test dataset is out-sample
        test_logits = MerlinUtil.generate_logits(self.shadow_model, self.v_test_loader, self.auxiliary_info.device)
        self.v_logits.append(test_logits)
        self.v_y.append(torch.zeros(test_logits.shape[0], dtype=torch.long, device=self.auxiliary_info.device))

        self.prepared = True

    def infer(self, target_data):
        """
        Infers the membership of target data with the Merlin attack.
        :param target_data: the target data.
        :return: the inferred membership.
        """
        target_x, target_y = target_data
        target_per_instance_loss = np.array(MerlinUtil.log_loss(target_y, self.target_model_access.model(target_x)))
        merlin_ratio = MerlinUtil.get_merlin_ratio(target_x, target_y, self.target_model_access.model,
                                                   target_per_instance_loss, self.noise_params,
                                                   self.auxiliary_info.max_t)

        thresh = MerlinUtil.get_inference_threshold(self.v_merlin_ratio, self.v_y,
                                                    self.auxiliary_info.attack_fpr_threshold)
        pred_membership = np.zeros(len(target_x))
        pred_membership[np.where(merlin_ratio > thresh)] = 1
        return pred_membership
