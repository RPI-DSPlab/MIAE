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
import re
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
from utils import load_target_dataset


# --------------------- functions and classes for ensemble -------------------------


class MetaModel(nn.Module):
    """MetaModel for stacking"""

    def __init__(self, input_dim: int):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)  # Adding dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def mia_ensemble_stacking(base_preds: List[prediction.Predictions], gt: np.array):
    """
    This function ensembles the predictions of the base models using stacking

    :param base_preds: List of predictions of the base models
    :param gt: Ground truth labels
    """

    num_base_model = len(base_preds)
    meta_model = MetaModel(input_dim=num_base_model)
    # Ensure the numpy arrays are converted to torch.float32 tensors
    meta_features = torch.stack([torch.from_numpy(p.pred_arr).float() for p in base_preds],
                                dim=1)  # stack all predictions from attacks
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


def mia_ensemble_avg(base_preds: List[prediction]):
    """
    This function ensembles the predictions of the base models using averaging

    :param base_preds: List of predictions of the base models
    """

    num_base_model = len(base_preds)
    num_samples = base_preds[0].pred_arr.shape[0]
    ensemble_preds = np.zeros_like(base_preds[0].pred_arr, dtype=np.float32)
    for p in base_preds:
        ensemble_preds += p.pred_arr.astype(np.float32)
    ensemble_preds /= num_base_model

    return ensemble_preds


# --------------------- helping functions -----------------------------------------

def read_pred(preds_path: str, extend_name: str, sd: int, dataset: str, model: str,
              attack: str, gt: np.ndarray) -> prediction.Predictions:
    """
    Read the prediction file and return them, the format of prediction follows: f"preds_sd{seed}{extend_name}"

    :param preds_path: the path to the predictions folders
    :param extend_name: the extension of the name of the file (what goes after preds_sd{seed})
    :param sd: seed to obtain the predictions
    :param dataset: the dataset used for the predictions
    :param model: the model used for the predictions
    :param attack: the attack used for the predictions
    :param gt: ground true of the membership prediction
    """

    pred_file = os.path.join(preds_path, f"preds_sd{sd}{extend_name}", dataset, model, attack, f"pred_{attack}.npy")
    pred = np.load(pred_file)
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    return prediction.Predictions(pred, gt, name=f"{dataset}_{model}_{attack}")


def read_preds(preds_path: str, extend_name: str, sds: List[int], dataset: str, model: str,
               attacks: List[str], gt: np.ndarray) -> List[List[prediction.Predictions]]:
    """
    wrapper function to read multiple predictions
    file directory is organized as: seed -> dataset -> model -> attack -> preds_{attack}.pkl

    :param preds_path: the path to the predictions folders
    :param extend_name: the extension of the name of the file (what goes after preds_sd{seed})
    :param sds: list of seed to obtain the predictions
    :param dataset: dataset used for the predictions
    :param model: model used for the predictions
    :param attacks: list of attack used for the predictions
    :param gt: ground true of the membership prediction

    :return: List of predictions
    """

    ret_list = []
    for sd in sds:
        list_x = []
        for attack in attacks:
            list_x.append(read_pred(preds_path, extend_name, sd, dataset, model, attack, gt))
        ret_list.append(list_x)

    return ret_list


def prepare_ensemble(preds_on_shadow_target: List[prediction.Predictions], gt, ensemble_method: str, save_path: str):
    """
    prepare_ensemble prepares a ensemble and save the ensemble model if needed

    preds_on_shadow_target: list of predictions on shadow-target model
    gt: ground truth labels
    ensemble_method: method to ensemble the predictions [avg, stacking]
    save_path: path to save the ensemble model
    """
    if ensemble_method == "avg":
        print("NO need to prepare for avg")
    elif ensemble_method == "stacking":
        meta_model = mia_ensemble_stacking(preds_on_shadow_target, gt)
        torch.save(meta_model, save_path + "/stacking_meta_model.pth")
    else:
        raise ValueError("Invalid ensemble method")


def run_ensemble(base_preds: List[prediction.Predictions], dataset_to_attack: Dataset, ensemble_method: str,
                 save_path: str, ensemble_save_path: str):
    """
    run the ensemble on the dataset_to_attack

    base_preds: list of predictions to be ensemble
    dataset_to_attack: dataset to attack
    ensemble_method: method to ensemble the predictions [avg, stacking]
    save_path: path to save the ensemble result
    ensemble_save_path: path to saved ensemble file (ie: meta model)
    """

    if ensemble_method == "avg":
        ensemble_preds = mia_ensemble_avg(base_preds)
    elif ensemble_method == "stacking":
        meta_model = torch.load(ensemble_save_path + "/stacking_meta_model.pth")
        meta_features = torch.stack([torch.from_numpy(p.pred_arr).float() for p in base_preds], dim=1)
        ensemble_preds = meta_model(meta_features).detach().numpy()
    else:
        raise ValueError("Invalid ensemble method")

    # save the ensemble result
    with open(save_path + f"/ensemble_preds_{ensemble_method}.pkl", "wb") as f:
        if ensemble_method == "stacking":
            pickle.dump(np.transpose(ensemble_preds)[1], f)
        else:
            pickle.dump(ensemble_preds, f)



def get_ensemble_methods(directory):
    files = os.listdir(directory)
    methods = []
    fns = []

    # Regex pattern to match the ensemble method names
    pattern = re.compile(r'ensemble_preds_(.+)\.pkl')

    for file in files:
        # Match the pattern to extract the method name
        match = pattern.match(file)
        if match:
            methods.append(match.group(1))
            fns.append(file)

    return methods, fns


def multi_seed_avg(pred_list: List[List[prediction.Predictions]]) -> List[prediction.Predictions]:
    """
    Average the predictions across multiple seeds

    :param pred_list: List of predictions from multiple seeds that follows the structure of read_preds
    :return: List of averaged predictions
    """

    avg_preds = []
    for i in range(len(pred_list[0])):
        pred_attack_i = []
        for j in range(len(pred_list)):
            pred_attack_i.append(pred_list[j][i])
        avg_preds.append(prediction.multi_seed_ensemble(pred_attack_i, "avg"))

    return avg_preds


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
    parser.add_argument('--target_model_path', type=str, help='same as shadow_save_path')
    parser.add_argument('--data_aug', type=bool, default=False, help='whether to use data augmentation')
    parser.add_argument('--attack_lr', type=float, default=0.1, help='learning rate for MIA training')
    parser.add_argument('--attack_epochs', type=int, default=100, help='number of epochs for MIA training')
    parser.add_argument('--target_epochs', type=int, default=100, help='number of epochs for target model training')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument("--dataset", type=str, default="cifar10", help='dataset: [cifar10, cifar100]')
    # ----- prepare ensemble arguments ----
    parser.add_argument('--preds_path', type=str,
                        help='Save path for the predictions (both shadow preds and base preds)')
    parser.add_argument("--ensemble_seeds", type=int, nargs="+", help="Random seed")
    parser.add_argument("--attacks", type=str, nargs="+", default=None,
                        help='MIA type: [losstraj, yeom, shokri, aug, lira]')
    parser.add_argument("--ensemble_save_path", type=str, help="Path to save the ensemble files (ie: meta model)")
    parser.add_argument('--shadow_target_data_path', type=str,
                        help='Save path for shadow target data, ground truth, etc')
    parser.add_argument('--ensemble_method', type=str, help='Ensemble method [avg, stacking]')
    # ---- run ensemble arguments ----
    parser.add_argument('--target_data_path', type=str, help='path to target data')
    parser.add_argument('--ensemble_result_path', type=str, help='path to save the ensemble pred result')
    args = parser.parse_args()

    set_seed(args.seed)
    if args.mode == "train_shadow":  # train shadow-target model
        with open(os.path.join(args.aux_set_path, "aux_set.pkl"), "rb") as f:
            original_aux_set = pickle.load(f)

        # re-partition the auxiliary set
        target_set_len = int(len(original_aux_set) / 2)
        target_set, aux_set = dataset_utils.dataset_split(original_aux_set,
                                                          [target_set_len, len(original_aux_set) - target_set_len])
        target_train_set, target_test_set = dataset_utils.dataset_split(target_set, [int(0.5 * target_set_len),
                                                                                     int(0.5 * target_set_len)])

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
        target_model = train_target_model(target_model, args.shadow_save_path, args.device, target_train_set,
                                          target_test_set, args)

        # save datasets
        dataset_save_path = os.path.join(args.shadow_save_path, f"{args.dataset}")
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)
        with open(os.path.join(dataset_save_path, "target_trainset.pkl"), "wb") as f:
            pickle.dump(target_train_set, f)
        with open(os.path.join(dataset_save_path, "target_testset.pkl"), "wb") as f:
            pickle.dump(target_test_set, f)
        with open(os.path.join(dataset_save_path, "aux_set.pkl"), "wb") as f:
            pickle.dump(aux_set, f)
        index_to_data = {}
        for i in range(len(dataset_to_attack)):
            index_to_data[i] = dataset_to_attack[i]

        with open(os.path.join(dataset_save_path, "index_to_data.pkl"), "wb") as f:
            pickle.dump(index_to_data, f)

        # save the membership
        np.save(os.path.join(dataset_save_path, "attack_set_membership.npy"), target_membership)

    elif args.mode == "train_ensemble":
        # read the predictions
        pred_path = args.preds_path

        # read data
        with open(os.path.join(args.shadow_target_data_path, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(args.shadow_target_data_path, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(args.shadow_target_data_path, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)
        index_to_data, membership = load_target_dataset(args.shadow_target_data_path)

        shadow_target_preds = read_preds(pred_path, "_ensemble_base", args.ensemble_seeds, args.dataset,
                                         args.target_model,
                                         args.attacks, membership)

        # prepare the ensemble (single seed)
        preds_on_shadow_target_list = shadow_target_preds[0]  # getting preds from the 0th seeds' prediction
        save_path = os.path.join(args.ensemble_save_path, "single_seed")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        prepare_ensemble(preds_on_shadow_target_list, membership, args.ensemble_method, save_path)

        # prepare the ensemble (multi-seed)
        preds_on_shadow_target_list = multi_seed_avg(shadow_target_preds)
        save_path = os.path.join(args.ensemble_save_path, "multi_seed")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        prepare_ensemble(preds_on_shadow_target_list, membership, args.ensemble_method, save_path)

    elif args.mode == "run_ensemble":
        # read the predictions
        pred_path = args.preds_path
        base_preds = read_preds(pred_path, "", args.ensemble_seeds, args.dataset, args.target_model,
                                args.attacks, None)

        # read target data
        with open(os.path.join(args.target_data_path, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)

        # run the ensemble (single seed)
        base_preds_list = base_preds[0]
        dataset_to_attack = ConcatDataset([target_trainset, target_testset])
        ensemble_result_path = os.path.join(args.ensemble_result_path, "single_seed")
        if not os.path.exists(ensemble_result_path):
            os.makedirs(ensemble_result_path)
        run_ensemble(base_preds_list, dataset_to_attack, args.ensemble_method, ensemble_result_path,
                     args.ensemble_save_path + f"/{args.dataset}" + "/single_seed")

        # run the ensemble (multi seed)
        base_preds_list = multi_seed_avg(base_preds)
        ensemble_result_path_multi = os.path.join(args.ensemble_result_path, "multi_seed")
        if not os.path.exists(ensemble_result_path_multi):
            os.makedirs(ensemble_result_path_multi)
        run_ensemble(base_preds_list, dataset_to_attack, args.ensemble_method, ensemble_result_path_multi,
                     args.ensemble_save_path + f"/{args.dataset}" + "/multi_seed")


    elif args.mode == "evaluation":
        # read target data
        with open(os.path.join(args.target_data_path, "target_trainset.pkl"), "rb") as f:
            target_trainset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "target_testset.pkl"), "rb") as f:
            target_testset = pickle.load(f)
        with open(os.path.join(args.target_data_path, "aux_set.pkl"), "rb") as f:
            aux_set = pickle.load(f)
        index_to_data, membership = load_target_dataset(args.target_data_path)

        # read original preds
        pred_path = args.preds_path
        base_preds = read_preds(pred_path, "", args.ensemble_seeds, args.dataset, args.target_model,
                                args.attacks, membership)
        base_preds_list = base_preds[0]

        # read ensemble preds (single seed)
        ensemble_result_path = args.ensemble_result_path + "/single_seed"
        methods_names_single_seed, file_names = get_ensemble_methods(ensemble_result_path)
        methods_names_single_seed = ["single_seed_" + x for x in methods_names_single_seed]
        ensemble_preds_single_seed = []
        for fn in file_names:
            with open(os.path.join(ensemble_result_path, fn), "rb") as f:
                ensemble_preds_single_seed.append(prediction.Predictions(pickle.load(f), membership, fn))

        # read ensemble preds (multi seed)
        ensemble_result_path = args.ensemble_result_path + "/multi_seed"
        methods_names_multi_seed, file_names = get_ensemble_methods(ensemble_result_path)
        methods_names_multi_seed = ["multi_seed_" + x for x in methods_names_multi_seed]
        ensemble_preds_multi_seed = []
        for fn in file_names:
            with open(os.path.join(ensemble_result_path, fn), "rb") as f:
                ensemble_preds_multi_seed.append(prediction.Predictions(pickle.load(f), membership, fn))

        # combine single attack result with ensemble result
        name_list = args.attacks + methods_names_single_seed + methods_names_multi_seed
        preds_list = base_preds_list + ensemble_preds_single_seed + ensemble_preds_multi_seed

        # auc
        prediction.plot_auc(preds_list, name_list, f"{args.dataset} {args.target_model} ensemble AUC", save_path=args.ensemble_result_path + '/ensemble_roc.png')

        # printing some stats
        balanced_acc = []
        acc = []

        for attack_name, pred in zip(name_list, preds_list):
            acc.append(pred.accuracy())
            balanced_acc.append(pred.balanced_attack_accuracy())

        for i in range(len(name_list)):
            print(f"{name_list[i]}: \tacc: {acc[i]:.4f}, \tbalanced acc: {balanced_acc[i]:.4f}")
