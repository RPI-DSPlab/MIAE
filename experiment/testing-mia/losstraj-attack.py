import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
import numpy as np

from mia.attacks import losstraj_mia, base as mia_base
from utils.datasets.loader import load_dataset
from utils.set_seed import set_seed



dataset_dir = "datasets"
result_dir = "results"
configs_dir = "configs"
dataset_name = "cifar10"
batch_size = 1000
trainset_ratio = 0.5  # meaning 50% of the training set is used for training the target model


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
        x = x.view(-1, 64 * 8 * 8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main(testing=False):
    set_seed(0)

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

    training_set, shadow_set = random_split(trainset, [trainset_ratio * len(trainset),
                                                       len(trainset) - trainset_ratio * len(trainset)])

    # defining a target model
    """NOTE: again, we are allowing user to define their own target model"""
    target_model = TargetModel()

    # training the target model
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(100)):
        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = target_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Print batch-level loss and accuracy
            tqdm.write(
                f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}, Accuracy: {(correct_predictions / total_samples) * 100:.2f}%")



    # Calculate and print the accuracy at the end of the epoch
    accuracy = (correct_predictions / total_samples) * 100
    print(f"End of Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")

    # save the target model
    torch.save(target_model.state_dict(), os.path.join(result_dir, "target_model.pth"))

    # initialize the target model access
    target_model_access = losstraj_mia.LosstrajModelAccess(target_model, mia_base.ModelAccessType.WHITE_BOX)

    # initialize the attack
    merlin_attack_config = losstraj_mia.LosstrajAuxiliaryInfo(None)  # None would result in default config
    attack = losstraj_mia.LosstrajAttack(target_model_access, merlin_attack_config)

    attack.prepare(shadow_set)

    # attack the target model
    attack_test = ConcatDataset([training_set, testset])
    y_membership = np.concatenate((np.ones(len(training_set)), np.zeros(len(training_set))))

    pred_membership = attack.infer(attack_test)

    # calculate the accuracy
    accuracy = np.sum(y_membership == pred_membership) / len(y_membership)
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main(testing=False)
