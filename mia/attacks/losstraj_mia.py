# This code implements the loss trajectory based membership inference attack, published in CCS 2022 "Membership
# Inference Attacks by Exploiting Loss Trajectory".
# The code is based on the code from
# https://github.com/DennisLiu2022/Membership-Inference-Attacks-by-Exploiting-Loss-Trajectory

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack

class LosstrajAuxiliaryInfo(AuxiliaryInfo):
    """
    The auxiliary information for the loss trajectory based membership inference attack.
    """
    def __init__(self, config: dict):
        """
        Initialize the auxiliary information.
        :param config: the loss trajectory.
        """
        super().__init__(config)
        self.loss_trajectory = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')


class LosstrajModelAccess(ModelAccess):
    """
    The model access for the loss trajectory based membership inference attack.
    """
    def __init__(self, model, model_type):
        """
        Initialize the model access.
        :param model: the target model.
        :param model_type: the type of the target model.
        """
        super().__init__(model, model_type)

    def get_loss_trajectory(self, input_signals):
        """
        Get the loss trajectory of the target model.
        :param input_signals: the input signals.
        :return: the loss trajectory.
        """

class LosstrajUtil:

    @classmethod
    def load_dataset(cls, x, y, aug: bool, train: bool) -> DataLoader:
        """
        load the dataset with given x and y and return the dataloader.
        """
    @classmethod
    def build_trajectory_membership_dataset(cls, args, target_model, dataset, device):
        MODEL = target_model.to(device)


        if args.mode == 'target':
            print('load target_dataset ... ')
            train_loader = dataset.aug_target_train_loader
            test_loader = dataset.aug_target_test_loader

        elif args.mode == 'shadow':
            print('load shadow_dataset ... ')
            train_loader = dataset.aug_shadow_train_loader
            test_loader = dataset.aug_shadow_test_loader

        model_top1 = None
        model_loss = None
        orginal_labels = None
        predicted_labels = None
        predicted_status = None
        member_status = None

        def normalization(data):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

        MODEL.eval()

        for loader_idx, data_loader in enumerate([train_loader, test_loader]):
            top1 = DATA.AverageMeter()
            for data_idx, (data, target, ori_idx) in enumerate(data_loader):
                batch_trajectory = get_trajectory(data, target, args, ori_model_path, device)
                data, target = data.to(device), target.to(device)
                batch_logit_target = MODEL(data)

                _, batch_predict_label = batch_logit_target.max(1)
                batch_predicted_label = batch_predict_label.long().cpu().detach().numpy()
                batch_original_label = target.long().cpu().detach().numpy()
                batch_loss_target = [F.cross_entropy(batch_logit_target_i.unsqueeze(0), target_i.unsqueeze(0)) for
                                     (batch_logit_target_i, target_i) in zip(batch_logit_target, target)]
                batch_loss_target = np.array(
                    [batch_loss_target_i.cpu().detach().numpy() for batch_loss_target_i in batch_loss_target])
                batch_predicted_status = (
                            torch.argmax(batch_logit_target, dim=1) == target).float().cpu().detach().numpy()
                batch_predicted_status = np.expand_dims(batch_predicted_status, axis=1)
                member = np.repeat(np.array(int(1 - loader_idx)), batch_trajectory.shape[0], 0)
                batch_loss_ori = batch_loss_target

                model_loss_ori = batch_loss_ori if loader_idx == 0 and data_idx == 0 else np.concatenate(
                    (model_loss_ori, batch_loss_ori), axis=0)
                model_trajectory = batch_trajectory if loader_idx == 0 and data_idx == 0 else np.concatenate(
                    (model_trajectory, batch_trajectory), axis=0)
                original_labels = batch_original_label if loader_idx == 0 and data_idx == 0 else np.concatenate(
                    (original_labels, batch_original_label), axis=0)
                predicted_labels = batch_predicted_label if loader_idx == 0 and data_idx == 0 else np.concatenate(
                    (predicted_labels, batch_predicted_label), axis=0)
                predicted_status = batch_predicted_status if loader_idx == 0 and data_idx == 0 else np.concatenate(
                    (predicted_status, batch_predicted_status), axis=0)
                member_status = member if loader_idx == 0 and data_idx == 0 else np.concatenate((member_status, member),
                                                                                                axis=0)

        print(f'------------Loading trajectory {args.mode} dataset successfully!---------')
        data = {
            'model_loss_ori': model_loss_ori,
            'model_trajectory': model_trajectory,
            'original_labels': original_labels,
            'predicted_labels': predicted_labels,
            'predicted_status': predicted_status,
            'member_status': member_status,
            'nb_classes': dataset.num_classes
        }

        dataset_type = 'trajectory_train_data' if args.mode == 'shadow' else 'trajectory_test_data'
        utils.create_path(ori_model_path + f'/{args.mode}/{model_name}')
        np.save(ori_model_path + f'/{args.mode}/{model_name}/{dataset_type}', data)


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

    def prepare(self, shadow_set):
        """
        Prepare the attack.
        """
