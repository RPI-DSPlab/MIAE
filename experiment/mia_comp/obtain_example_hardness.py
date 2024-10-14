"""
This script is outdated. We no longer conduct this type of analysis.
This script is used to obtain the example hardness on a given dataset.
Work flow:
1. Load the dataset
2. Obtain the example hardness
3. Save the example hardness

Note that this script is to help with the analysis of obtain_pred, therefore it will directly
load the dataset (specifically, the target dataset) that's saved by `obtain_pred.py`.
"""

import argparse
import os
import pickle

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import logging
import sys

sys.path.append(os.path.join(os.getcwd(), "..", ".."))

from miae.utils.set_seed import set_seed
# from miae.sample_metrics import iteration_learned, prediction_depth, consistency_score
# from miae.sample_metrics.sample_metrics_config import iteration_learned_config, prediction_depth_config, \
#     consistency_score_config
from miae.sample_metrics import iteration_learned, prediction_depth
from miae.sample_metrics.sample_metrics_config import iteration_learned_config, prediction_depth_config

from experiment import models


class IndexedDataset(Dataset):
    """
    Dataset Wrapper to return the index of the example as well.
    """
    def __init__(self, dataset: Dataset, num_classes: int = None):
        self.dataset = dataset
        self.num_classes = num_classes

    def __getitem__(self, idx):
        return idx, self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_num_classes(self):
        return self.num_classes



def load_dataset(file_path: str) -> Dataset:
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def train_example_hardness(example_hardness: str, dataset: Dataset, args, model):
    """
    Train the example hardness on the given dataset.
    """
    if example_hardness == "il":
        config = iteration_learned_config.IterationLearnedConfig(
            {"num_epochs": 100, "seeds": [0, 1, 2, 3], "save_path": args.preparation_path,
             "log_path": args.result_path})
        example_hardness = iteration_learned.IlHardness(config, model, dataset)
        example_hardness.train_metric()
    elif example_hardness == "pd":
        config = prediction_depth_config.PredictionDepthConfig(
            {"num_epochs": 100, "seeds": [0, 1, 2, 3], "save_path": args.preparation_path,
             "log_path": args.result_path})
        example_hardness = prediction_depth.PdHardness(config, model, dataset)
        example_hardness.train_metric()
    elif example_hardness == "cs":
        pass
    else:
        raise ValueError("Invalid example hardness type")

    return example_hardness

def plot_hist(data, title, xlabel, ylabel, save_path, bins=20):
    """
    This function is used to plot a histogram of the data.
    :param data: the data to be plotted
    :param title: the title of the plot
    :param xlabel: the label of the x-axis
    :param ylabel: the label of the y-axis
    :param save_path: the path to save the plot
    :param bins: the number of bins in the histogram
    :return: None
    """
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Obtain the example hardness on a given dataset")

    # mandatory arguments
    parser.add_argument("--example_hardness", type=str, required=True, help="[il, pd]")
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the datasets")
    parser.add_argument('--model', type=str, default=None,
                        help='target model arch: [resnet56, wrn32_4, vgg16, mobilenet]')
    parser.add_argument('--result_path', type=str, default=None, help='path to save the prediction')
    parser.add_argument('--preparation_path', type=str, default=str(os.getcwd()), help='path to the preparation files')
    parser.add_argument('--dataset', type=str, default=None, help='dataset: [cifar10, cifar100, cinic10]')

    # optional arguments (eg. training hyperparameters)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for example hardness training')
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='device to train the model')

    args = parser.parse_args()

    # set the random seed
    set_seed(args.seed)

    # create all the necessary directories
    for path in [args.result_path, args.preparation_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # load dataset
    trainset = load_dataset(os.path.join(args.dataset_path, f"{args.dataset}/target_trainset.pkl"))
    testset = load_dataset(os.path.join(args.dataset_path, f"{args.dataset}/target_testset.pkl"))
    fullset = ConcatDataset([trainset, testset])

    num_classes = None
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif args.dataset == "cinic10":
        num_classes = 10
    else:
        raise ValueError("Invalid dataset")

    fullset = IndexedDataset(fullset, num_classes)  # wrap the dataset to return the index of the example as well

    # initialize model
    input_size = None
    if args.dataset == "cifar10":
        input_size = 32
    elif args.dataset == "cifar100":
        input_size = 32
    elif args.dataset == "cinic10":
        input_size = 32
    else:
        raise ValueError("Invalid dataset")

    model = models.get_model(args.model, num_classes, input_size)

    print(f"obtaining {args.example_hardness} on {args.dataset} with {args.model} model")
    # obtain the example hardness
    example_hardness = train_example_hardness(args.example_hardness, fullset, args, model)

    # save the example hardness
    scores = []
    id_list = [int(idx) for idx, _ in fullset]
    for i in id_list:
        scores.append(example_hardness.get_score(i))
    scores = np.array(scores)
    with open(os.path.join(args.result_path, f"{args.example_hardness}_score.pkl"), "wb") as f:
        pickle.dump(scores, f)

    # plot the histogram of the scores
    plot_hist(scores, f"{args.example_hardness} score", "score", "frequency",
              os.path.join(args.result_path, f"{args.example_hardness}_score_hist.png"), bins=args.epochs+1)
