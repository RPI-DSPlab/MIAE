import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from typing import List
import numpy as np
from tqdm import tqdm
from copy import deepcopy

# add mia to path
import sys
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

from mia.utils.set_seed import set_seed
from mia.attacks import losstraj_mia, merlin_mia
from mia.attacks import base as mia_base
from mia.utils import roc_auc

batch_size = 1000
trainset_ratio = 0.5  # percentage of training set to be used for training the target model
train_test_ratio = 0.9  # percentage of training set to be used for training any model that uses a test set

target_train_epochs = 200

current_dir = os.getcwd()
target_model_dir = os.path.join(current_dir,"target_model")
attack_dir = os.path.join(current_dir,"attack")
savedir = os.path.join(current_dir,"results")
seed = 0



class TargetModel(torch.nn.Module):
    """
    a basic target model for CIFAR10
    """

    def __init__(self):
        super(TargetModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_target_model(target_model_dir: str, device: torch.device, trainset: Dataset, testset: Dataset):
    """
    train a target model and save to target_model_dir

    :param target_model_dir: directory to save target model
    :param device: device to train target model
    :param trainset: training set (member)
    :param testset: test set (non-member)
    """

    target_model = TargetModel().to(device)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

    correct_predictions = 0
    total_samples = 0

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

        if epoch % 20 == 0 or epoch == target_train_epochs - 1:
            target_model.eval()
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = target_model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            print(f"Epoch {epoch} accuracy: {correct_predictions / total_samples} loss: {loss.item()}")

    # save the target model
    torch.save(target_model.state_dict(), os.path.join(target_model_dir, "target_model.pth"))


def obtain_roc_auc(attacks: List[mia_base.MiAttack], savedir: str, data, membership: np.ndarray):
    """
    obtain roc and auc for  given (prepared) attacks and save to savedir
    :param attacks: a list of prepared attacks
    :param savedir: directory to save roc and auc
    :param data: data to predict membership
    :param membership: membership of data
    :return: None
    """

    attack_pred_save_dir = os.path.join(savedir, "attack_predictions")
    if not os.path.exists(attack_pred_save_dir):
        os.makedirs(attack_pred_save_dir)

    for attack in attacks:
        if attack.prepared is False:
            raise ValueError(f"{attack.__class__.__name__} is not prepared")

    predictions = []
    attacks_names = []
    for attack in attacks:
        if not os.path.exists(os.path.join(attack_pred_save_dir, attack.__class__.__name__+'_pred.npy')):
            prediction = attack.infer(data)
            np.save(os.path.join(attack_pred_save_dir, attack.__class__.__name__+'_pred.npy'), prediction)
        else:
            prediction = np.load(os.path.join(attack_pred_save_dir, attack.__class__.__name__+'_pred.npy'))
        predictions.append(prediction)
        attacks_names.append(attack.__class__.__name__)

    membership_list = [membership for _ in range(len(attacks))]

    print(predictions[0], predictions[0].shape)

    roc_auc.fig_fpr_tpr(predictions, membership_list, attacks_names,savedir)


def main():
    # set up
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dir in [target_model_dir, attack_dir, savedir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # initialize the dataset
    train_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                ])
    test_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                ])

    # load cifar10
    """NOTE: I am loading cifar10 from torchvision instead of using datasets/loader.py because 
    users are supposed to be able to use """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # prepare the shadow set and target set and then train a target model
    dataset = ConcatDataset([trainset, testset])
    train_len = int(len(dataset) * trainset_ratio)
    shadow_len = len(dataset) - train_len
    training_set, aux_set = random_split(dataset, [train_len, shadow_len])
    training_set, aux_set = training_set.dataset, aux_set.dataset

    target_trainset, target_testset = random_split(dataset, [int(len(dataset) * train_test_ratio),
                                               len(dataset) - int(len(dataset) * train_test_ratio)])

    # -- STEP 1: train target model
    if not os.path.exists(os.path.join(target_model_dir, "target_model.pth")):
        train_target_model(target_model_dir, device, target_trainset, target_testset)

    target_model = TargetModel()
    target_model.load_state_dict(torch.load(os.path.join(target_model_dir, "target_model.pth")))

    # -- STEP 2: prepare the attacks
    losstraj_aux_info = losstraj_mia.LosstrajAuxiliaryInfo(
            {'device': device, 'seed': seed, 'save_path': attack_dir+'/losstraj'})
    merlin_aux_info = merlin_mia.MerlinAuxiliaryInfo(
            {'device': device, 'seed': seed, 'save_path': attack_dir+'/merlin'})

    losstraj_target_model_access = losstraj_mia.LosstrajModelAccess(deepcopy(target_model))
    merlin_target_model_access = merlin_mia.MerlinModelAccess(deepcopy(target_model))

    attacks = [
        losstraj_mia.LosstrajAttack(losstraj_target_model_access, losstraj_aux_info),
        merlin_mia.MerlinAttack(merlin_target_model_access, merlin_aux_info)
    ]

    # -- prepare the attacks
    for attack in attacks:
        print("Preparing attack: ", attack.__class__.__name__)
        attack.prepare(aux_set)

    # -- STEP 3: attack the target dataset
    # mix the target trainset and testset and then attack
    target_dataset = ConcatDataset([target_trainset, target_testset])
    target_membership = np.concatenate([np.ones(len(target_trainset)), np.zeros(len(target_testset))])

    obtain_roc_auc(attacks, savedir, target_dataset, target_membership)



if __name__ == "__main__":
    main()


        

