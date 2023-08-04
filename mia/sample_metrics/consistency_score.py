import copy
from abc import ABC

from tqdm import tqdm
from mia.sample_metrics.base import ExampleMetric
import torch
import os
import numpy as np
import json

from utils import models as smmodels
from utils import datasets as smdatasets
from sample_metrics.sample_metrics_config.consistency_score_config import ConsistencyScoreConfig


class CSHardness(ExampleMetric, ABC):
    """Compute the hardness of a dataset using the consistency score metric."""

    def __init__(self, config: ConsistencyScoreConfig, model: smmodels, dataset: smdatasets):
        """
        Initialize the ConsistencyScoreMetric instance by providing a configuration object, a model, and a dataset.
        :param config: A ConsistencyScoreConfig object.
        :param model: A model object, which should be an instance of the utils.models
        :param dataset: A dataset object, which should be an instance of the utils.datasets
        """
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.ready = False  # flag to indicate whether the model has been trained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_train_total = len(self.dataset.train_set)
        self.cscores = None

        if config.crit == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Criterion {} not implemented".format(config.crit))

        if config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9,
                                             weight_decay=5e-4)
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(config.optimizer))

        # create save path
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        self.save_path = self.config.save_path

    def train_metric(self):
        """
        Train the model for the consistency score metric, then save the result.
        """
        "inner functions are helper functions for `train_metric()`"
        "--------------------------------------------------------------------"

        def subset_train(seed, device, subset_ratio, config):
            torch.manual_seed(seed)
            np.random.seed(seed)

            num_train_total = len(self.dataset.train_set)
            num_train = int(num_train_total * subset_ratio)

            indices = torch.randperm(num_train_total)

            train_subset_indices = indices[:num_train]  # used in training
            # held-out example, we get c-scores on this
            test_subset_indices = indices[num_train:]

            empty_indices = torch.tensor([], dtype=torch.long)
            train_loader, _ = self.dataset.subset_loaders(config.batch_size, train_subset_indices, empty_indices, False,
                                                          False, num_workers=1)  # we only need the train loader
            test_loader, _ = self.dataset.subset_loaders(config.batch_size, test_subset_indices, empty_indices, False,
                                                         False, num_workers=1)  # we only need the train loader

            # copy the model
            model = copy.deepcopy(self.model)
            model.to(device)

            training_acc = []

            for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
                for imgs, labels, idx in train_loader:
                    imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                    self.optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for (imgs, labels), idx in train_loader:
                        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                        outputs = model(imgs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    training_acc.append(correct / total)
                model.train()

            trainset_mask = torch.zeros(num_train_total, dtype=torch.bool)
            trainset_mask[test_subset_indices] = True

            trainset_correctness = {}

            model.eval()
            with torch.no_grad():
                for idx in test_subset_indices:
                    trainset_correctness[idx.item()] = 0
                for (imgs, labels), idx in test_loader:
                    imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    for i in range(len(predicted)):
                        if predicted[i] == labels[i]:
                            trainset_correctness[idx[i].item()] = 1

            subset_acc = sum(trainset_correctness.values()) / len(trainset_correctness)
            print(f"Subset Accuracy: {subset_acc:.4f}")
            print(f"Training Accuracy: {training_acc[-1]:.4f}")
            return trainset_correctness

        "--------------------------------------------------------------------"

        # def subset_train
        n_runs = self.config.n_runs  # number of runs
        ss_ratio = self.config.ss_ratio  # subset ratio

        results = []
        for i_run in range(n_runs):
            print(f'Run {i_run + 1}/{n_runs} ----------------------------------')
            results.append(subset_train(self.config.seed, self.device, ss_ratio, self.config))

        train_rep = {}  # number of times each image is predicted in the loop above
        train_correctness_sum = {}  # sum of correctly prediction for each image in the loop above
        for i_run in tqdm(range(n_runs), desc=f'calculate c-scores'):
            for idx in results[i_run]:  # for each image's prediction's correctness
                if idx not in train_rep:
                    train_rep[idx] = 0
                    train_correctness_sum[idx] = 0
                train_rep[idx] += 1
                if results[i_run][idx]:
                    train_correctness_sum[idx] += 1

        cscores = {}

        avg_train_rep_list = []
        for idx in train_rep:
            cscores[idx] = train_correctness_sum[idx] / train_rep[idx]
            avg_train_rep_list.append(train_rep[idx])
        avg_train_rep = sum(avg_train_rep_list) / len(avg_train_rep_list)

        print(
            f"-----\n the average number of times each image is predicted is {avg_train_rep}\nwith min={min(avg_train_rep_list)} and max={max(avg_train_rep_list)}\n-----")

        # save the result
        with open(os.path.join(self.save_path, "cs_run{}_{}_trainratio{}_train_avg.json".format(self.config.n_runs,
                                                                                                str(self.dataset),
                                                                                                self.config.ss_ratio)),
                  "w") as f:
            json.dump(cscores, f)

        self.cscores = cscores
        self.ready = True

    def load_metric(self, path):
        """
        Load the consistency score metric from a file.
        :param path: The path to the file containing the consistency score metric.
        :return:
        """
        with open(path, "r") as f:
            self.cscores = json.load(f)
        self.ready = True

    def get_score(self, idx: int, train: bool = True):
        """
        Get the consistency score of an example.
        :param idx: The index of the example to get the consistency score of.
        :param train: Whether to get the consistency score of the training set or the testing set.
        :return: The consistency score of the example.
        """
        if not self.ready:
            raise ValueError("ConsistencyScoreMetric not ready. Call train_metric() or load_metric() first.")
        if train == False:
            raise NotImplementedError("ConsistencyScoreMetric only supports train=True.")
        return self.cscores[idx]

    def __repr__(self):
        """
        String representation of the ConsistencyScoreMetric class.
        :return: String representation of the ConsistencyScoreMetric class.
        """
        return "ConsistencyScoreMetric({})".format(self.__dict__)

    def __str__(self):
        """
        String representation of the ConsistencyScoreMetric class.
        :return: String representation of the ConsistencyScoreMetric class.
        """
        return self.__repr__()
