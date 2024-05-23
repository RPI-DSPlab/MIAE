"""
This file consist classes and functions that are used to ensemble
attacks, and scripts to read attack results and ensemble them.
We may later move the ensemble functions and classes to MIAE
framework.

what does this file do?
1. train shadow-target model based on pre-saved auxiliary set
2. prepare ensemble: train the ensemble on the prediction on shadow-target model
3. ensemble the attack results
"""

import argparse
import os
import pickle
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import List, Dict
import numpy as np

import miae.eval_methods.sample_hardness

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
import miae.eval_methods.prediction as prediction
from miae.utils.set_seed import set_seed
from miae.utils import dataset_utils
from experiment import models
from obtain_pred import train_target_model



# --------------------- functions and classes for ensemble -------------------------


class MetaModel(nn.Module):
    """MetaModel for stacking"""
    def __init__(self, input_dim: int):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def mia_ensemble_stacking(base_preds: List[prediction], gt: np.array):
    """
    This function ensembles the predictions of the base models using stacking

    :param base_preds: List of predictions of the base models
    :param gt: Ground truth labels
    """

    num_base_model = len(base_preds)
    meta_model = MetaModel(input_dim=num_base_model)
    meta_features = torch.stack([p.pred_arr for p in base_preds], dim=1)  # stack all predictions from attacks
    meta_labels = torch.tensor(gt, dtype=torch.long)

    # Train the meta model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = meta_model(meta_features)
        loss = criterion(outputs, meta_labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total = meta_labels.size(0)
            correct = (predicted == meta_labels).sum().item()
            print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {100 * correct / total}")

    return meta_model





# --------------------- scripts for each mode of this file  -------------------------




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble the attack results')
    parser.add_argument('--mode', type=str, help='[train_shadow, train_ensemble, run_ensemble]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to train the model')
    # ----- train_shadow arguments -----
    parser.add_argument('--shadow_model', type=str, help='Shadow model to train')
    parser.add_argument('--aux_set_path', type=str, help='auxiliary set path')
    parser.add_argument('--shadow_save_path', type=str, help='Save path for shadow model')
    parser.add_argument('--target_model', type=str, default=None,
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument('--data_aug', type=bool, default=False, help='whether to use data augmentation')
    parser.add_argument('--attack_lr', type=float, default=0.1, help='learning rate for MIA training')
    parser.add_argument('--attack_epochs', type=int, default=100, help='number of epochs for MIA training')
    parser.add_argument('--target_epochs', type=int, default=100, help='number of epochs for target model training')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "train_shadow":  # train shadow-target model
        with open(os.path.join(args.aux_set_path, "aux_set.pkl"), "rb") as f:
            original_aux_set = pickle.load(f)

        # re-partition the auxiliary set
        target_set_len = int(len(original_aux_set) / 2)
        target_set, aux_set = dataset_utils.dataset_split(original_aux_set, [target_set_len, len(original_aux_set) - target_set_len])
        target_train_set, target_test_set = dataset_utils.dataset_split(target_set, [int(0.5 * target_set_len), int(0.5 * target_set_len)])

        dataset_to_attack = ConcatDataset([target_train_set, target_test_set])
        target_membership = np.concatenate([np.ones(len(target_train_set)), np.zeros(len(target_test_set))])

        # training the target model
        if args.dataset == "cifar10":
            num_classes = 10
            input_size = 32
        elif args.dataset == "cifar100":
            num_classes = 100
            input_size = 32
        else:
            raise ValueError("Invalid dataset")
        target_model = models.get_model(args.target_model, num_classes, input_size).to(args.device)
        target_model = train_target_model(target_model, args.shadow_save_path, args.device, target_train_set, target_test_set, args)

        # save target model
        torch.save(target_model.state_dict(), os.path.join(args.shadow_save_path, "target_model.pth"))

        # save datasets
        dataset_save_path = os.path.join(args.shadow_save_path, f"{args.dataset}")
        with open(os.path.join(dataset_save_path, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(dataset_save_path, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(dataset_save_path, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)
        index_to_data = {}
        for i in range(len(dataset_to_attack)):
            index_to_data[i] = dataset_to_attack[i]

        with open(os.path.join(dataset_save_path, "index_to_data.pkl"), "wb") as f:
            pickle.dump(index_to_data, f)

        # save the membership
        np.save(os.path.join(dataset_save_path, "attack_set_membership.npy"), target_membership)
