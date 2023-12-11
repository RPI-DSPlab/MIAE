import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
import numpy as np

from mia.attacks import merlin_mia, base as mia_base
from utils.datasets.loader import load_dataset
from utils.set_seed import set_seed



dataset_dir = "datasets"
result_dir = "results"
configs_dir = "configs"
dataset_name = "cifar10"
batch_size = 10000
trainset_ratio = 0.5  # meaning 50% of the training set is used for training the target model
shadow_split_ratio = 0.8  # meaning 20% of the shadow set is used for training the shadow model
num_epochs = 250



# define a model architecture
class TargetModel(torch.nn.Module):
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
        x = x.reshape(-1, 64 * 8 * 8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main(testing=False):
    set_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # intialize the directories
    for d in [dataset_dir, result_dir, configs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # initialize the dataset
    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
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
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    # prepare the shadow set and target set and then train a target model
    train_len = int(len(trainset) * trainset_ratio)
    shadow_len = len(trainset) - train_len
    training_set, shadow_set = random_split(trainset, [train_len, shadow_len])

    # defining a target model

    target_model = TargetModel()

    if os.path.exists(os.path.join(result_dir, "target_model.pth")):
        target_model.load_state_dict(torch.load(os.path.join(result_dir, "target_model.pth")))
    else:
        # training the target model
        target_model.to(device)
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
        target_model.train()
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.005, weight_decay=0.0005)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for i, data in enumerate(training_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = target_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate the accuracy for this batch
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = (correct_predictions / total_samples) * 100
            if epoch % 20 == 0:
                print(f"Epoch {epoch + 1},\tAccuracy: {accuracy:.2f}%,\tLoss: {loss.item():.4f}")



        # Calculate and print the accuracy at the end of the epoch
        accuracy = (correct_predictions / total_samples) * 100
        print(f"End of Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")

        # save the target model
        torch.save(target_model.state_dict(), os.path.join(result_dir, "target_model.pth"))

    # initialize the target model access
    target_model_access = merlin_mia.MerlinModelAccess(target_model, mia_base.ModelAccessType.GRAY_BOX)

    # initialize the attack
    config = dict()
    merlin_attack_config = merlin_mia.MerlinAuxiliaryInfo(config)  # None would result in default config
    shadow_model = TargetModel()
    attack = merlin_mia.MerlinAttack(target_model_access, merlin_attack_config, shadow_model)

    # split shadow_set
    shadow_train_len = int(len(shadow_set) * shadow_split_ratio)
    shadow_test_len = len(shadow_set) - shadow_train_len
    shadow_training_set, shadow_test_set = random_split(shadow_set, [shadow_train_len, shadow_test_len])
    attack.prepare([shadow_training_set, shadow_test_set])

    # attack the target model
    attack_test = ConcatDataset([training_set, testset])
    y_membership = np.concatenate((np.ones(len(training_set)), np.zeros(len(testset))))
    pred_membership = attack.infer((attack_test, y_membership))

    # calculate the accuracy
    accuracy = np.sum(y_membership == pred_membership) / len(y_membership)
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main(testing=False)
