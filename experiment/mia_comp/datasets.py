"""
This script contains functions that loads different datasets
"""
import torchvision
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100, ImageFolder


def get_cifar10(aug:bool=True) -> ConcatDataset:
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


def get_cinic10(aug: bool = True) -> ConcatDataset:
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]
    regular_transform = T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor(),
                                   T.Normalize(mean=mean, std=std)
                                   ])

    augmentation_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.transforms.ToTensor(),
                                        T.Normalize(mean=mean, std=std)])

    transform = augmentation_transform if aug else regular_transform

    trainset = ImageFolder(root='./data/CINIC-10/train', transform=transform)

    testset = ImageFolder(root='./data/CINIC-10/test', transform=transform)

    return ConcatDataset([trainset, testset])
