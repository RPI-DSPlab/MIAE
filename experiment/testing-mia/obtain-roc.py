import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import List
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt

# add miae to path
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

from miae.utils.set_seed import set_seed
from miae.attacks import losstraj_mia, merlin_mia, lira_mia
from miae.attacks import base as mia_base
from miae.utils import roc_auc, dataset_utils
from experiment import models

batch_size = 128
targetset_ratio = 0.35  # percentage of training set to be used for training/test the target model
train_test_ratio = 0.5  # percentage of training set to be used for training any model that uses a test set
lr = 0.1
target_train_epochs = 100

current_dir = os.getcwd()
target_model_dir = os.path.join(current_dir,"target_model")
attack_dir = os.path.join(current_dir,"attack")
savedir = os.path.join(current_dir,"results")
seed = 0

aug = True


def print_key_stats(predictions: np.ndarray, ground_truth: np.ndarray, attack_name: str, savedir: str):
    """
    print key stats of predictions including mean, std, min, max, and shape, then plot the histogram of predictions
    :param predictions: predictions to print stats
    :param ground_truth: the answer of membership
    :param attack_name: name of the attack
    :param savedir: directory to save the histogram
    :return: None
    """

    # calculate accuracy of prediction
    _, one_hot_pred = torch.max(torch.tensor(predictions), 1)
    accuracy = np.mean(one_hot_pred == ground_truth)
    print(f"accuracy on testset of attack {attack_name}: {accuracy*100:.2f}%")

    plt.hist(predictions)
    plt.title(f"{attack_name} predictions")
    plt.savefig(os.path.join(savedir, f"{attack_name}_predictions_stats.png"))
    plt.close()


def train_target_model(model, target_model_dir: str, device: torch.device, trainset: Dataset, testset: Dataset):
    """
    train a target model and save to target_model_dir

    :param model: model to train
    :param target_model_dir: directory to save target model
    :param device: device to train target model
    :param trainset: training set (member)
    :param testset: test set (non-member)
    """

    target_model = model.to(device)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)

    # Create a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, target_train_epochs)

    print("Training target model")
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
            print(f"Epoch {epoch} train_acc: {train_correct_prediction / len(trainset):.2f} test_acc: {test_correct_predictions / len(testset):.2f} loss: {loss.item():.4f}， lr: {scheduler.get_last_lr()[0]:.4f}")

    # save the target model
    torch.save(target_model.state_dict(), os.path.join(target_model_dir, target_model.__class__.__name__ + "_target_model.pkl"))


def obtain_roc_auc(attacks: List[mia_base.MiAttack], savedir: str, data_to_attack: Dataset, membership: np.ndarray):
    """
    obtain roc and auc for  given (prepared) attacks and save to savedir
    :param attacks: a list of prepared attacks
    :param savedir: directory to save roc and auc
    :param data_to_attack: data to predict membership
    :param membership: membership of data
    :return: None
    """

    attack_pred_save_dir = os.path.join(savedir, "attack_predictions")
    if not os.path.exists(attack_pred_save_dir):
        os.makedirs(attack_pred_save_dir)

    for attack in attacks:
        if attack.prepared is False:
            raise ValueError(f"{attack.__class__.__name__} is not prepared")
        break  # SKIPPING MERLIN FOR NOW

    # save the index that's predicted
    dummy_data_loader = torch.utils.data.DataLoader(data_to_attack, batch_size=1, shuffle=False)
    id_list = []
    target_list = []
    for idx, (data, target) in enumerate(dummy_data_loader):
        id_list.append(idx)
        target_list.append(target)
    id_arr = np.array(id_list)
    target_arr = np.array(target_list)
    np.save(os.path.join(attack_pred_save_dir, 'id_of_prediction.npy'), id_arr)
    np.save(os.path.join(attack_pred_save_dir, 'target_of_prediction.npy'), target_arr)

    predictions = []
    attacks_names = []
    for attack in attacks:
        if not os.path.exists(os.path.join(attack_pred_save_dir, attack.__class__.__name__+'_pred.npy')):
            prediction = attack.infer(data_to_attack)
            np.save(os.path.join(attack_pred_save_dir, attack.__class__.__name__+'_pred.npy'), prediction)
        else:
            prediction = np.load(os.path.join(attack_pred_save_dir, attack.__class__.__name__+'_pred.npy'))
        predictions.append(prediction)
        attacks_names.append(attack.__class__.__name__)

    membership_list = [membership for _ in range(len(attacks))]

    roc_auc.fig_fpr_tpr(predictions, membership_list, attacks_names,savedir)


def main():
    # set up
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dir in [target_model_dir, attack_dir, savedir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # initialize the dataset
    regular_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=mean, std=std)
                                ])

    augmentation_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.transforms.ToTensor(), T.Normalize(mean=mean, std=std)])

    transform = augmentation_transform if aug else regular_transform

    # load cifar10
    """NOTE: I am loading cifar10 from torchvision instead of using datasets/loader.py because 
    users are supposed to be able to use """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # prepare the shadow set and target set and then train a target model
    dataset = ConcatDataset([trainset, testset])
    target_len = int(len(dataset) * targetset_ratio)
    shadow_len = len(dataset) - target_len
    target_set, aux_set = dataset_utils.dataset_split(dataset, [target_len, shadow_len])

    target_trainset, target_testset = dataset_utils.dataset_split(target_set, [int(len(target_set) * train_test_ratio),
                                               len(target_set) - int(len(target_set) * train_test_ratio)])


    # -- STEP 1: train target model
    target_model = models.create_wideresnet32_4()
    untrained_target_model = deepcopy(target_model)
    print("Target model: ", target_model.__class__.__name__, " is being trained with ", target_trainset.__class__.__name__, "len: ", len(target_trainset), " and ", target_testset.__class__.__name__, "len: ", len(target_testset))
    if not os.path.exists(os.path.join(target_model_dir, target_model.__class__.__name__ + "_target_model.pkl")):
        train_target_model(target_model, target_model_dir, device, target_trainset, target_testset)
    else:
        target_model.load_state_dict(torch.load(os.path.join(target_model_dir, target_model.__class__.__name__ + "_target_model.pkl")))

    # -- STEP 2: prepare the attacks
    losstraj_aux_info = losstraj_mia.LosstrajAuxiliaryInfo(
            {'device': device, 'seed': seed, 'save_path': attack_dir+'/losstraj', 'num_classes': 10, 'batch_size': batch_size, 'lr': lr, 'distillation_epochs': target_train_epochs})
    merlin_aux_info = merlin_mia.MerlinAuxiliaryInfo(
            {'device': device, 'seed': seed, 'save_path': attack_dir+'/merlin', 'num_classes': 10, 'batch_size': batch_size})
    lira_aux_info = lira_mia.LiraAuxiliaryInfo(
            {'device': device, 'seed': seed, 'save_path': attack_dir+'/lira', 'num_classes': 10, 'batch_size': batch_size, 'lr': lr, 'epochs': target_train_epochs, 'log_path': attack_dir+'/lira'})

    losstraj_target_model_access = losstraj_mia.LosstrajModelAccess(deepcopy(target_model), untrained_target_model)
    merlin_target_model_access = merlin_mia.MerlinModelAccess(deepcopy(target_model), untrained_target_model)
    lira_target_model_access = lira_mia.LiraModelAccess(deepcopy(target_model), untrained_target_model)

    attacks = [
        # losstraj_mia.LosstrajAttack(losstraj_target_model_access, losstraj_aux_info),
        # merlin_mia.MerlinAttack(merlin_target_model_access, merlin_aux_info),
        lira_mia.LiraAttack(lira_target_model_access, lira_aux_info)
    ]

    # -- prepare the attacks
    for attack in attacks:
        print("Preparing attack: ", attack.__class__.__name__)
        attack.prepare(aux_set)

    # -- STEP 3: attack the target dataset
    # mix the target trainset and testset and then attack
    dataset_to_attack = ConcatDataset([target_trainset, target_testset])
    target_membership = np.concatenate([np.ones(len(target_trainset)), np.zeros(len(target_testset))])

    obtain_roc_auc(attacks, savedir, dataset_to_attack, target_membership)



if __name__ == "__main__":
    main()




