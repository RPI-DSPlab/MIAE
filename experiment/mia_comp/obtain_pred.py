"""
This script is used to obtain the prediction of MIA on a specific target model and dataset.
Work flow:
1. Load the target model and dataset
2. Train the MIA model, all files generated during the training process will be saved in the preparation_path and get deleted after the training process
3. Obtain the prediction of MIA on the target model and dataset
4. Save the prediction and log in the result_path
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import List
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import logging
import pickle

# add miae to path
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

from miae.utils.set_seed import set_seed
from miae.attacks import losstraj_mia, shokri_mia, lira_mia, yeom_mia, aug_mia
from miae.attacks import base as mia_base
from miae.utils import roc_auc, dataset_utils
from experiment import models
from experiment.mia_comp import datasets


def get_dataset(datset_name, aug, targetset_ratio, train_test_ratio, shuffle_seed=1) -> tuple:
    """
    Get the datasets for the target model and MIA
    :param datset_name: name of the dataset
    :param aug: data augmentation when loading the dataset
    :param targetset_ratio: the ratio of the data used for target model training over the whole dataset
    :param train_test_ratio: the ratio of the data used for target model training over the target set
    :param shuffle_seed: seed for shuffling the dataset, default to 1
    :return:
    """
    if datset_name == "cifar10":
        dataset = datasets.get_cifar10(aug)
        num_classes = 10
        input_size = 32
    elif datset_name == "cifar100":
        dataset = datasets.get_cifar100(aug)
        num_classes = 100
        input_size = 32
    elif datset_name == "cinic10":
        dataset = datasets.get_cinic10(aug)
        num_classes = 10
        input_size = 32
    else:
        raise ValueError("Invalid dataset")

    # prepare the shadow set and target set
    target_len = int(len(dataset) * targetset_ratio)
    shadow_len = len(dataset) - target_len
    target_set, aux_set = dataset_utils.dataset_split(dataset, [target_len, shadow_len], shuffle_seed)

    target_trainset, target_testset = dataset_utils.dataset_split(target_set,
                                                                  [int(len(target_set) * train_test_ratio),
                                                                   len(target_set) - int(
                                                                       len(target_set) * train_test_ratio)], shuffle_seed)

    return target_trainset, target_testset, aux_set, num_classes, input_size


def train_target_model(model, target_model_dir: str, device: torch.device, trainset: Dataset, testset: Dataset, arg):
    """
    train a target model and save to target_model_dir

    :param model: model to train
    :param target_model_dir: directory to save target model
    :param device: device to train target model
    :param trainset: training set (member)
    :param testset: test set (non-member)
    :param arg: parsed command line argument containing training hyperparameters
    """

    batch_size = arg.batch_size
    target_train_epochs = arg.target_epochs
    lr = arg.attack_lr

    target_logger = logging.getLogger('target_logger')
    target_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(target_model_dir, f"target_model_{arg.target_model}_{arg.dataset}.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    target_logger.addHandler(fh)

    # log the target dataset size
    target_logger.info(f"Target dataset size: {len(trainset)}")

    target_model = model.to(device)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=lr, momentum=0.9,
                                weight_decay=0.0001)

    # Create a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, target_train_epochs)

    print("Training target model")
    logging.info("Training target model")
    for epoch in tqdm(range(target_train_epochs)):
        target_model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = target_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == target_train_epochs - 1:
            target_model.eval()
            test_correct_predictions = 0
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = target_model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_correct_predictions += (predicted == labels).sum().item()

            train_correct_prediction = 0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = target_model(inputs)
                _, predicted = torch.max(outputs, 1)
                train_correct_prediction += (predicted == labels).sum().item()
            target_logger.info(
                f"Epoch {epoch} train_acc: {train_correct_prediction / len(trainset):.2f} test_acc: {test_correct_predictions / len(testset):.2f} loss: {loss.item():.4f}， lr: {scheduler.get_last_lr()[0]:.4f}")

    # save the target model
    torch.save(target_model.state_dict(),
               os.path.join(args.target_model_path, "target_model_" + args.target_model + args.dataset + ".pkl"))


def get_target_model_access(args, target_model, untrained_target_model) -> mia_base.ModelAccess:
    """
    get the target model access for specified MIA
    :param args: parsed command line argument
    :param target_model: target model (trained)
    :param untrained_target_model: target model (untrained)
    :return: target model access
    """
    if args.attack == "losstraj":
        return losstraj_mia.LosstrajModelAccess(deepcopy(target_model), untrained_target_model)
    if args.attack == "yeom":
        return yeom_mia.YeomModelAccess(deepcopy(target_model), untrained_target_model)
    if args.attack == "shokri":
        return shokri_mia.ShokriModelAccess(deepcopy(target_model), untrained_target_model)
    if args.attack == "lira":
        return lira_mia.LiraModelAccess(deepcopy(target_model), untrained_target_model)
    if args.attack == "aug":
        return aug_mia.AugModelAccess(deepcopy(target_model), untrained_target_model)
    else:
        raise ValueError("Invalid attack type")


def get_aux_info(args, device: str, num_classes: int) -> mia_base.AuxiliaryInfo:
    """
    get the auxiliary information for specified MIA
    :param args: parsed command line argument
    :param device: device to run the MIA
    :param num_classes: number of classes in the dataset
    :return: auxiliary information
    """
    if args.attack == "losstraj":
        return losstraj_mia.LosstrajAuxiliaryInfo(
            {'device': device, 'seed': args.seed, 'save_path': args.preparation_path, 'num_classes': num_classes,
             'batch_size': args.batch_size, 'lr': 0.1, 'distillation_epochs': args.attack_epochs,
             'log_path': args.result_path})
    if args.attack == "yeom":
        return yeom_mia.YeomAuxiliaryInfo(
            {'device': device, 'seed': args.seed, 'save_path': args.preparation_path, 'num_classes': num_classes,
             'batch_size': args.batch_size, 'lr': 0.1, 'epochs': args.attack_epochs, 'log_path': args.result_path})
    if args.attack == "shokri":
        return shokri_mia.ShokriAuxiliaryInfo(
            {'device': device, 'seed': args.seed, 'save_path': args.preparation_path, 'num_classes': num_classes,
             'batch_size': args.batch_size, 'lr': 0.1, 'epochs': args.attack_epochs, 'log_path': args.result_path})
    if args.attack == "lira":
        return lira_mia.LiraAuxiliaryInfo(
            {'device': device, 'seed': args.seed, 'save_path': args.preparation_path, 'num_classes': num_classes,
             'batch_size': args.batch_size, 'lr': 0.1, 'epochs': args.attack_epochs, 'log_path': args.result_path})
    if args.attack == "aug":
        return aug_mia.AugAuxiliaryInfo(
            {'device': device, 'seed': args.seed, 'save_path': args.preparation_path, 'num_classes': num_classes,
             'batch_size': args.batch_size, 'lr': 0.1, 'epochs': args.attack_epochs, 'log_path': args.result_path})
    else:
        raise ValueError("Invalid attack type")


def get_attack(args, aux_info: mia_base.AuxiliaryInfo, target_model_access: mia_base.ModelAccess) -> mia_base.MiAttack:
    """
    get the attack for specified MIA
    :param args: parsed command line argument
    :param aux_info: auxiliary information for this attack
    :param target_model_access: target model access for this attack
    :return: attack
    """

    if args.attack == "losstraj":
        return losstraj_mia.LosstrajAttack(target_model_access, aux_info)
    if args.attack == "yeom":
        return yeom_mia.YeomAttack(target_model_access, aux_info)
    if args.attack == "shokri":
        return shokri_mia.ShokriAttack(target_model_access, aux_info)
    if args.attack == "lira":
        return lira_mia.LiraAttack(target_model_access, aux_info)
    if args.attack == "aug":
        return aug_mia.AugAttack(target_model_access, aux_info)
    else:
        raise ValueError("Invalid attack type")


def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obtain_membership_inference_prediction')

    # special arguments
    """if this argument is not non, the script will only save the dataset and exit. This is used to make sure the
    index - data mapping is consistent across different runs. This is useful when we want to compare the performance."""
    parser.add_argument('--save_dataset', type=bool, default=False, help='whether to save the dataset')
    parser.add_argument('--train_target_model', type=bool, default=False, help='whether to train the target model')

    # mandatory arguments
    parser.add_argument('--attack', type=str, default=None, help='MIA type: [losstraj, yeom, shokri ,lira, aug]')
    parser.add_argument('--target_model', type=str, default=None,
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument('--dataset', type=str, default=None, help='dataset: [cifar10, cifar100, cinic10]')
    parser.add_argument('--result_path', type=str, default=None, help='path to save the prediction')
    parser.add_argument('--data_path', type=str, default=None, help='path to the dataset')
    parser.add_argument('--target_model_path', type=str, default=str(os.getcwd()), help='path to the target model')
    parser.add_argument('--preparation_path', type=str, default=str(os.getcwd()), help='path to the preparation file')
    parser.add_argument('--target_set_ratio', type=float, default=0.5,
                        help='the ratio of the data used for target model training over the whole dataset')
    parser.add_argument('--train_test_ratio', type=float, default=0.5, help='train test ratio for target and MIA')
    parser.add_argument('--delete-files', type=bool, default=True,
                        help='whether to delete the preparation files after training')
    parser.add_argument('--shuffle_seed', type=int, default=1, help='seed for shuffling the dataset')

    # optional arguments (eg. training hyperparameters)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--data_aug', type=bool, default=False, help='whether to use data augmentation')
    parser.add_argument('--attack_lr', type=float, default=0.1, help='learning rate for MIA training')
    parser.add_argument('--attack_epochs', type=int, default=100, help='number of epochs for MIA training')
    parser.add_argument('--target_epochs', type=int, default=100, help='number of epochs for target model training')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='device to train the model')

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # create all the necessary directories
    for path in [args.result_path, args.target_model_path, args.preparation_path, args.data_path]:
        if path is not None and not os.path.exists(path):
            os.makedirs(path)

    if args.save_dataset:  # save the dataset and exit
        # initialize the dataset
        target_trainset, target_testset, aux_set, num_classes, input_size = get_dataset(args.dataset, args.data_aug,
                                                                                        args.target_set_ratio,
                                                                                        args.train_test_ratio, args.shuffle_seed)
        dataset_save_path = os.path.join(args.data_path, f"{args.dataset}")
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)

        # Save using pickle
        with open(os.path.join(dataset_save_path, "target_trainset.pkl"), "wb") as f:
            pickle.dump(target_trainset, f)
        with open(os.path.join(dataset_save_path, "target_testset.pkl"), "wb") as f:
            pickle.dump(target_testset, f)
        with open(os.path.join(dataset_save_path, "aux_set.pkl"), "wb") as f:
            pickle.dump(aux_set, f)

        # concat the target trainset and testset and then attack
        dataset_to_attack = ConcatDataset([target_trainset, target_testset])
        target_membership = np.concatenate([np.ones(len(target_trainset)), np.zeros(len(target_testset))])

        # Save the index - data mapping
        index_to_data = {}
        for i in range(len(dataset_to_attack)):
            index_to_data[i] = dataset_to_attack[i]

        with open(os.path.join(dataset_save_path, "index_to_data.pkl"), "wb") as f:
            pickle.dump(index_to_data, f)

        # save the membership
        np.save(os.path.join(dataset_save_path, "attack_set_membership.npy"), target_membership)

        exit(0)

    dataset_save_path = os.path.join(args.data_path, f"{args.dataset}")
    # load the dataset
    with open(os.path.join(dataset_save_path, "target_trainset.pkl"), "rb") as f:
        target_trainset = pickle.load(f)
    with open(os.path.join(dataset_save_path, "target_testset.pkl"), "rb") as f:
        target_testset = pickle.load(f)
    with open(os.path.join(dataset_save_path, "aux_set.pkl"), "rb") as f:
        aux_set = pickle.load(f)

    _, _, _, num_classes, input_size = get_dataset(args.dataset, args.data_aug,
                                                   args.target_set_ratio,
                                                   args.train_test_ratio, args.shuffle_seed)
    dataset_to_attack = ConcatDataset([target_trainset, target_testset])
    target_membership = np.concatenate([np.ones(len(target_trainset)), np.zeros(len(target_testset))])

    # training the target model
    target_model_copy = models.get_model(args.target_model, num_classes, input_size).to(args.device)
    target_model = models.get_model(args.target_model, num_classes, input_size).to(args.device)
    if args.train_target_model:  # we are only training the target model
        train_target_model(target_model, args.target_model_path, args.device, target_trainset, target_testset, args)
        exit(0)
    else:
        if not os.path.exists(
                os.path.join(args.target_model_path, "target_model_" + args.target_model + args.dataset + ".pkl")):
            raise ValueError(
                f'Target model does not exist at {os.path.join(args.target_model_path, "target_model_" + args.target_model + args.dataset + ".pkl")}')
        target_model.load_state_dict(
            torch.load(
                os.path.join(args.target_model_path, "target_model_" + args.target_model + args.dataset + ".pkl")))
        target_model.eval()

    # prepare the attack
    target_model_access = get_target_model_access(args, target_model, target_model_copy)
    aux_info = get_aux_info(args, args.device, num_classes)
    attack = get_attack(args, aux_info, target_model_access)
    attack.prepare(aux_set)

    # obtain the prediction
    pred = attack.infer(dataset_to_attack)
    np.save(os.path.join(args.result_path, "pred_" + args.attack + ".npy"), pred)

    # print the accuracy
    print(f"Accuracy: {np.mean((pred > 0.5) == target_membership):.4f}")

