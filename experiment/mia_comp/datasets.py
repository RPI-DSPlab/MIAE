"""
This script contains functions that loads different datasets
"""
import torchvision
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100, ImageFolder
from torchvision import transforms

class CINIC10(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img, label = self.image_folder[idx]
        return (self.transform(img), label)

class PurchaseTexas100(Dataset):
    def __init__(self, features, labels):

        self.labels = labels
        self.features = features
        # calculates the number of classes, samples, and features
        self.n_classes = len(np.unique(self.labels))
        self.n_samples = len(self.features)
        self.n_features = self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label



def get_cifar10(aug: bool = True) -> ConcatDataset:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    regular_transform = T.Compose([T.ToTensor(),
                                   T.Normalize(mean=mean, std=std)
                                   ])

    augmentation_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.transforms.ToTensor(),
                                        T.Normalize(mean=mean, std=std)])

    transform = augmentation_transform if aug else regular_transform

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    return ConcatDataset([trainset, testset])


def get_cifar100(aug: bool = True) -> ConcatDataset:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    regular_transform = T.Compose([T.ToTensor(),
                                   T.Normalize(mean=mean, std=std)
                                   ])

    augmentation_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.transforms.ToTensor(),
                                        T.Normalize(mean=mean, std=std)])

    transform = augmentation_transform if aug else regular_transform

    trainset = CIFAR100(root='./data', train=True,
                        download=True, transform=transform)

    testset = CIFAR100(root='./data', train=False,
                       download=True, transform=transform)

    return ConcatDataset([trainset, testset])

def get_cinic10(image_dir, aug: bool = True) -> ConcatDataset:
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]
    regular_transform = T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor(),
                                   T.Normalize(mean=mean, std=std)
                                   ])
    augmentation_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.transforms.ToTensor(),
                                        T.Normalize(mean=mean, std=std)])
    transform = augmentation_transform if aug else regular_transform
    trainset = ImageFolder(root=f'{image_dir}/CINIC10_60ksubset/cifar10_train_subset', transform=transform)
    testset = ImageFolder(root=f'{image_dir}/CINIC10_60ksubset/imagenet_test_subset', transform=transform)
    return ConcatDataset([trainset, testset])


def get_purchase_texas(dataset, path_to_data='./data', subset_size=60000, dtype=torch.float32) -> Dataset:
    """
    load purchase100 or texas100 dataset from
    https://github.com/xehartnort/Purchase100-Texas100-datasets?tab=readme-ov-file

    dataset: 'purchase100' or 'texas100'
    path_to_data: path to the data folder to download or load
    subset_size: number of samples to load from the dataset, None means to load the whole dataset
    """

    # find the path to the dataset
    if not os.path.exists(f'{path_to_data}/purchase100-texas100/{dataset}.npz'):
        if not os.path.exists(f'{path_to_data}/purchase100-texas100'):
            os.makedirs(f'{path_to_data}/purchase100-texas100')
        print(f"Donwloading {dataset} dataset to {path_to_data}/purchase100-texas100")
        os.system(f'git clone https://github.com/xehartnort/Purchase100-Texas100-datasets.git {path_to_data}/purchase100-texas100')

    data = np.load(f'{path_to_data}/purchase100-texas100/{dataset}.npz')


    features = data['features']
    labels = data['labels']
    # make labels one-hot
    labels = np.argmax(labels, axis=1)

    if subset_size is not None:
        subset_indices = np.random.choice(len(features), subset_size, replace=False)
        features = features[subset_indices]
        labels = labels[subset_indices]

    features = torch.tensor(features, dtype=dtype)
    labels = torch.tensor(labels, dtype=torch.long)

    purchase_dataset = PurchaseTexas100(features, labels)

    return purchase_dataset


def get_texas100(path_to_data='./data') -> Dataset:
    return get_purchase_texas('texas100', path_to_data)

def get_purchase100(path_to_data='./data') -> Dataset:
    return get_purchase_texas('purchase100', path_to_data)
