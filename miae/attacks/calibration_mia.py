# This code implements the Loss Attack with Difficalty Calibration in the paper  On the Importance of Difficulty
# Calibration in Membership Inference Attacks
# The code is based on the code from
# https://github.com/facebookresearch/calibration_membership/tree/e2fde52aa67833bdba0f97a9a3b55a7500a273c9
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, ConcatDataset
from sklearn.metrics import roc_curve
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack, MIAUtils
from miae.utils.set_seed import set_seed
from miae.utils.dataset_utils import dataset_split
from miae.attacks.shokri_mia import ShokriUtil # for logging and splitting dataset


class CalibrationAuxiliaryInfo(AuxiliaryInfo):
    """
    The auxiliary information for the Calibration attack.
    """

    def __init__(self, config, attack_model=None):
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
        # note that aux model in this attack is equivalent to the definition of shadow model
        self.num_aux = config.get("num_aux", 1)  # number of auxiliary model, aux model is the model g in the paper
        self.num_shadow_epochs = config.get("epochs", 100)
        self.num_shadow_models = config.get("num_shadow_models", 1) 
        self.momentum = config.get("momentum", 0.9)
        self.weight_decay = config.get("weight_decay", 0.0001)
        self.num_classes = config.get("num_classes", 10)
        self.shadow_train_ratio = config.get("shadow_train_ratio", 0.5)

        # -- other parameters --
        self.shadow_diff_init = config.get("shadow_diff_init", True)  # different initialization for shadow models
        self.save_path = config.get("save_path", "calibration")
        self.shadow_model_path = config.get("shadow_model_path", f"{self.save_path}/shadow_models")
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

        # if log_path is None, no log will be saved, otherwise, the log will be saved to the log_path
        self.log_path = config.get('log_path', None)
        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.log_path is not None:
            self.logger = logging.getLogger('calibration_logger')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_path + '/calibration.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)


class CalibrationModelAccess(ModelAccess):
    """
    Implementation of model access for Calibration attack.
    """

    def __init__(self, model, untrained_model, access_type: ModelAccessType = ModelAccessType.BLACK_BOX):
        """
        Initialize model access with model and access type.
        :param model: the target model.
        :param access_type: the type of access to the target model.
        """
        super().__init__(model, untrained_model, access_type)


class CalibrationUtil(MIAUtils):

    @classmethod
    def get_loss(cls, dataset: Dataset, model, device="cpu") -> np.ndarray:
        """
        Get the loss of the dataset from the model.

        dataset: the dataset to get the loss (data, label).
        model: the model to calculate the loss.
        device: the device model and data shall be on

        return: the loss of each dataset the dataset.
        """

        model.eval()
        model.to(device)
        loss = []
        for data, label in tqdm(dataset, desc="Calculating loss"):
            data = data.unsqueeze(0)
            label = torch.tensor([label]).to(device)
            output = model(data.to(device))
            loss.append(F.cross_entropy(output, label).item())

        return np.array(loss)


class CalibrationAttack(MiAttack):
    """
    Implementation of the Calibration attack.
    """

    def __init__(self, target_model_access: CalibrationModelAccess, aux_info: CalibrationAuxiliaryInfo, target_data=None):
        """
        Initialize the Calibration attack with model access and auxiliary information.
        :param target_model_access: the model access to the target model.
        :param aux_info: the auxiliary information for the Shokri attack.
        :param target_data: the target data for the Shokri attack.
        """
        super().__init__(target_model_access, aux_info)
        self.aux_info = aux_info
        self.target_model_access = target_model_access
        self.shadow_model = None  # this is the shadow model
        self.shadow_models = [] # this is the list of shadow models
        self.threshold = None  # this is the loss threshold for the attack

        self.prepared = False  # this flag indicates whether the attack has been prepared

    def prepare(self, auxiliary_dataset):
        """
        Prepare the attack: train a shadow model to obtain the loss threshold
        :param auxiliary_dataset: the auxiliary dataset (will be split into training sets)
        """
        super().prepare(auxiliary_dataset)
        if self.prepared:
            print("The attack has already prepared!")
            return

        set_seed(self.aux_info.seed)

        # create directories:
        for path in [self.aux_info.save_path, self.aux_info.log_path, self.aux_info.shadow_model_path]:
            if path is not None and not os.path.exists(path):
                os.makedirs(path)
        train_set_len = int(len(auxiliary_dataset) * self.aux_info.shadow_train_ratio)
        test_set_len = len(auxiliary_dataset) - train_set_len
        train_set, test_set = dataset_split(auxiliary_dataset, [train_set_len, test_set_len])

        CalibrationUtil.log(self.aux_info, "Start preparing the attack...", print_flag=True)

        if self.aux_info.num_shadow_models == 1:
            # train the shadow model
            self.shadow_model = self.target_model_access.untrained_model
            if os.path.exists(self.aux_info.save_path + '/shadow_model.pth'):
                self.shadow_model = torch.load(self.aux_info.save_path + '/shadow_model.pth')
            else:
                trainloader = DataLoader(train_set, batch_size=self.aux_info.batch_size, shuffle=True, num_workers=2)
                testloader = DataLoader(test_set, batch_size=self.aux_info.batch_size, shuffle=False, num_workers=2)

                try:
                    set_seed(self.aux_info.seed)
                    self.shadow_model.initialize_weights()
                except:
                    raise NotImplementedError("the model doesn't have .initialize_weights method")
                
                self.shadow_model = CalibrationUtil.train_shadow_model(self.shadow_model, trainloader, testloader, self.aux_info)
                torch.save(self.shadow_model, self.aux_info.shadow_model_path + '/shadow_model.pth')

        else:
            # creating non-overlapping shadow datasets
            sub_shadow_dataset_list = ShokriUtil.split_dataset(train_set, self.aux_info.num_shadow_models)
            # log/print the shadow dataset sizes
            ShokriUtil.log(self.aux_info, f"Shadow dataset[0] size: {sub_shadow_dataset_list[0].__len__()}")
            for i in range(self.aux_info.num_shadow_models):
                # train k shadow (reference) models 
                model_name = f"calibration_shadow_model_{i}.pt"
                model_path = os.path.join(self.aux_info.shadow_model_path, model_name)

                shadow_model_i = self.target_model_access.get_untrained_model()
                shadow_model_i.to(self.aux_info.device)

                if self.aux_info.shadow_diff_init:
                    try:
                        set_seed((self.aux_info.seed + i)*100) # *100 to avoid overlapping of different instances
                        shadow_model_i.initialize_weights()
                    except:
                        raise NotImplementedError("the model doesn't have .initialize_weights method")

                train_len = int(len(sub_shadow_dataset_list[i]) * self.aux_info.shadow_train_ratio)
                test_len = len(sub_shadow_dataset_list[i]) - train_len
                shadow_train_dataset, shadow_test_dataset = torch.utils.data.random_split(sub_shadow_dataset_list[i],
                                                                                          [train_len, test_len])

                shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=self.aux_info.batch_size,
                                                 shuffle=True)
                shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=self.aux_info.batch_size,
                                                shuffle=False)
                if os.path.exists(model_path):
                    ShokriUtil.log(self.aux_info,
                                   f"Loading shadow model {i + 1}/{self.aux_info.num_shadow_models}...")
                    shadow_model_i.load_state_dict(torch.load(model_path))
                else:
                    ShokriUtil.log(self.aux_info,
                                   f"Training shadow model {i + 1}/{self.aux_info.num_shadow_models}...")
                    shadow_model_i = ShokriUtil.train_shadow_model(shadow_model_i, shadow_train_loader,
                                                                   shadow_test_loader,
                                                                   self.aux_info)
                    torch.save(shadow_model_i.state_dict(), model_path)
                
                self.shadow_models.append(shadow_model_i)


        # get the loss values of aux set (public set) from shadow model (auxiliary model) and target model
        recombined_aux_set = ConcatDataset([train_set, test_set])  # make sure the order aligned with membership is correct
        if self.aux_info.num_shadow_models == 1:
            shadow_model_loss = CalibrationUtil.get_loss(recombined_aux_set, self.shadow_model, self.aux_info.device)
        else:
            shadow_model_loss = []
            for shadow_model in self.shadow_models:
                shadow_model_loss.append(CalibrationUtil.get_loss(recombined_aux_set, shadow_model, self.aux_info.device))
            shadow_model_loss = np.mean(shadow_model_loss, axis=0)

        target_model_loss = CalibrationUtil.get_loss(recombined_aux_set, self.target_model_access.model, self.aux_info.device)

        # now target model acts as the calibration model, and shadow model acts as the target model for mimic the real membership inference attack
        calibrated_loss = shadow_model_loss - target_model_loss
        membership_label = np.concatenate([np.ones(len(train_set)), np.zeros(len(test_set))])

        # calculate the threshold
        thresholds = np.linspace(min(calibrated_loss), max(calibrated_loss), 1000)
        accuracies = []
        for threshold in thresholds:
            pred = calibrated_loss > threshold
            accuracies.append(accuracy_score(membership_label, pred))

        self.threshold = thresholds[np.argmax(accuracies)]

        self.prepared = True
        CalibrationUtil.log(self.aux_info, "Finish preparing the attack...", print_flag=True)

    def infer(self, target_data) -> np.ndarray:
        """
        Infer the membership of the target data.
        """
        super().infer(target_data)
        if not self.prepared:
            raise ValueError("The attack has not been prepared!")
        losses_threshold_diff = []

        CalibrationUtil.log(self.aux_info, "Start membership inference...", print_flag=True)

        # load models
        self.target_model_access.to_device(self.aux_info.device)
        target_model_loss = CalibrationUtil.get_loss(target_data, self.target_model_access, self.aux_info.device)
        if self.aux_info.num_shadow_models == 1:
            shadow_model_loss = CalibrationUtil.get_loss(target_data, self.shadow_model, self.aux_info.device)
        else:
            shadow_model_loss = []
            for shadow_model in self.shadow_models:
                shadow_model_loss.append(CalibrationUtil.get_loss(target_data, shadow_model, self.aux_info.device))
            shadow_model_loss = np.mean(shadow_model_loss, axis=0)
            
        calibrated_loss = target_model_loss - shadow_model_loss
        losses_threshold_diff = calibrated_loss - self.threshold

        # for the purpose of obtaining the prediction as a score, we couldn't just use the boolean value of the
        # losses_threshold_diff, but we need to normalize the value to [0, 1]
        min_diff, max_diff = min(losses_threshold_diff), max(losses_threshold_diff)
        losses_threshold_diff = np.array(losses_threshold_diff)
        losses_threshold_diff = (losses_threshold_diff - min_diff) / (max_diff - min_diff)

        predictions = 1 - losses_threshold_diff

        CalibrationUtil.log(self.aux_info, "Finish membership inference...", print_flag=True)

        return predictions
