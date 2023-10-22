import os
import torch
import torchvision.transforms as T

from mia.attacks import merlin_mia
from utils.datasets.loader import load_dataset

dataset_dir = "datasets"
result_dir = "results"
configs_dir = "configs"
dataset_name = "cifar10"

def main(testing=False):
    # initialize the dataset
    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                 ])
    test_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                ])
    
    dataset = load_dataset(dataset_name, dataset_dir, train_transform, test_transform, target_transform)



if __name__ == "__main__":
    main(testing=False)
