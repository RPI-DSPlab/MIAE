import collections
import copy
import logging
import time
from abc import ABC
from tqdm import tqdm
import sys
import torch
import os
import numpy as np
import json
from torch.utils.data import DataLoader

from miae.sample_metrics.base import ExampleMetric
from miae.sample_metrics.sample_metrics_config.iteration_learned_config import IterationLearnedConfig
from miae.sample_metrics import sm_util


class IlHardness(ExampleMetric, ABC):
    """Computer the hardness of a dataset based on the iteration learned method."""

    def __init__(self, config: IterationLearnedConfig, model, dataset):
        """
        :param config: the configuration file
        :param model: the model
        :param dataset: the dataset to train and obtain iteration learned metric
        """
        super().__init__()
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = None  # the trainloader for the dataset
        self.ready = False  # whether the trainloader and testloader are ready
        self.optimizer = None
        self.model = model

        self.trainloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

        self.save_path = config.save_path  # the path to save the results
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(os.path.join(self.save_path, "results")):
            os.makedirs(os.path.join(self.save_path, "results"))
        self.result_file_path = os.path.join(self.save_path, "results")

        if not os.path.exists(os.path.join(self.save_path, "results_avg")):
            os.makedirs(os.path.join(self.save_path, "results_avg"))
        self.result_file_path_avg = os.path.join(self.save_path, "results_avg")

        self.model = model
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)


        # set up logging
        self.log = False
        if config.log_path is not None:
            self.log = True
            self.il_logger = logging.getLogger('iteration_learned_logger')
            self.il_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.config.log_path + '/il.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.il_logger.addHandler(fh)

    def _evaluate_model(self, model, dataloader, learning_history_dict):
        model.eval()
        with torch.no_grad():
            for idx, (imgs, labels) in dataloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                for i in range(len(idx)):
                    learning_history_dict[idx[i].item()].append(labels[i].item() == predicted[i].item())

    def _trainer(self, trainloader, model, optimizer, criterion, device,
                 learning_history_train_dict, config: IterationLearnedConfig):
        curr_iteration = 0
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs)
        history = {"train_loss": [], "train_acc": [], "lr": []}
        print('------ Training started on {} with total number of {} epochs ------'.format(device, config.num_epochs))
        if self.log:
            self.il_logger.info(
                '------ Training started on {} with total number of {} epochs ------'.format(device, config.num_epochs))
        for epoch in tqdm(range(config.num_epochs)):
            # time each epoch
            train_acc = 0
            train_loss = 0
            for idx, (imgs, labels) in trainloader:
                model.train()
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item()
                train_loss += loss.item()
                curr_iteration += 1
                if config.learned_metric == "iteration_learned":  # if we are using iteration learned metric
                    self._evaluate_model(model, trainloader, learning_history_train_dict)

            cos_scheduler.step()
            train_acc /= len(trainloader.dataset)
            train_loss /= len(trainloader.dataset)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if config.learned_metric == "epoch_learned":  # if we are using epoch learned metric
                self._evaluate_model(model, trainloader, learning_history_train_dict)

            if epoch % 20 == 0 or epoch == config.num_epochs - 1:
                print(
                    "Epoch: {} \ttrain_loss: {:.4f} \ttrain_acc: {:.2f} \t lr: {:.2f}"
                    .format(epoch, train_loss, train_acc, cos_scheduler.get_last_lr()[0]))
                if self.log:
                    self.il_logger.info(
                        "Epoch: {} \ttrain_loss: {:.4f} \ttrain_acc: {:.2f} \t lr: {:.2f}"
                        .format(epoch, train_loss, train_acc, cos_scheduler.get_last_lr()[0]))

    def _determine_learned_metric(self, learning_history_dict):
        """
        This function determines the learned metric for the model, it finds the iteration or epoch which the model has
        learned a datapoint
        :param learning_history_dict: dictionary containing the learning history of the model, if -1, then the model can't learn
            this datapoint
        """
        learned_metric_dict = {}
        for i in learning_history_dict:
            learned_itr = 0
            learned_bool = False
            curr_itr = 0
            for j in learning_history_dict[i]:
                if j == True:  # if the model has learned this datapoint
                    if learned_bool == False:
                        learned_itr = curr_itr
                    learned_bool = True
                else:  # if the model has not learned this datapoint or has forgotten it
                    learned_bool = False
                curr_itr += 1

            if learned_bool == True:
                learned_metric_dict[i] = learned_itr
            else:
                learned_metric_dict[i] = -1

        return learned_metric_dict

    def train_metric(self):
        """
        This function trains the model and determines the learned metric
        """
        # train the model
        seeds = self.config.seeds

        for seed in seeds:
            model_copy = copy.deepcopy(self.model)

            # set up the criterion and optimizer
            if self.config.crit == 'cross_entropy':
                self.criterion = torch.nn.CrossEntropyLoss()
            else:
                raise NotImplementedError("Criterion {} not implemented".format(self.config.crit))

            if self.config.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(model_copy.parameters(), lr=self.config.lr,
                                                 momentum=self.config.momentum,
                                                 weight_decay=self.config.weight_decay)
            else:
                raise NotImplementedError("Optimizer {} not implemented".format(self.config.optimizer))

            sm_util.set_seed(seed)
            model_copy = model_copy.to(self.device)

            learning_history_train_dict = {}
            for idx, _ in self.trainloader:
                for i in idx:
                    learning_history_train_dict[i.item()] = list()
            # train the model
            self._trainer(self.trainloader, model_copy, self.optimizer, self.criterion, self.device,
                          learning_history_train_dict, self.config)

            learned_metric_train = self._determine_learned_metric(learning_history_train_dict)

            # save the partial results
            if self.config.learned_metric == "iteration_learned":
                with open(os.path.join(self.result_file_path,
                                       "{}-{}-learned_metric_iteration_seed{}_train.json".format(self.dataset.__class__,
                                                                                             self.model.__class__,
                                                                                             seed)), "w") as f:
                    json.dump(learned_metric_train, f)
            elif self.config.learned_metric == "epoch_learned":
                with open(os.path.join(self.result_file_path,
                                       "{}-{}-learned_metric_epoch_seed{}_train.json".format(self.dataset.__class__,
                                                                                             self.model.__class__,
                                                                                             seed)), "w") as f:
                    json.dump(learned_metric_train, f)
            else:
                raise NotImplementedError("Learned metric {} not implemented".format(self.config.learned_metric))

        # average the results
        try:
            self.train_avg_score = sm_util.avg_result(self.result_file_path, file_suf="train.json")
        except Exception as e:
            raise Exception("No files found in the directory, please train the model first")

        # save the averaged results
        if self.config.learned_metric == "iteration_learned":
            with open(os.path.join(self.result_file_path_avg,
                                   "{}-{}-learned_metric_iteration_avg_train.json".format(self.dataset.__class__,
                                                                                             self.model.__class__)), "w") as f:
                json.dump(self.train_avg_score, f)
        elif self.config.learned_metric == "epoch_learned":
            with open(os.path.join(self.result_file_path_avg,
                                   "{}-{}-learned_metric_epoch_avg_train.json".format(self.dataset.__class__,
                                                                                             self.model.__class__)), "w") as f:
                json.dump(self.train_avg_score, f)
        self.ready = True

    def load_metric(self, path: str):
        """
        This function loads the learned metric from the saved files
        """
        try:
            self.train_avg_score = sm_util.avg_result(path, file_suf="train.json")
        except Exception as e:
            raise Exception("No files found in the directory, please train the model first")
        self.ready = True

    def get_score(self, idx: int):
        """
        This function returns the learned metric of a datapoint
        :param idx: the index of the datapoint
        :return: the learned metric of the datapoint
        """
        if not self.ready:
            raise Exception("Please train the model first")
        return self.train_avg_score[idx]


    def __repr__(self):
        ret = (f"Iteration learned metric:\n"
               f"model: {repr(self.model)}\n"
               f"dataset: {repr(self.dataset)}\n"
               f"config: {self.config}\n"
               f"ready: {self.ready}\n"
               )
        if self.ready:
            ret += (f"train_avg_score: {self.train_avg_score}\n"
                    f"test_avg_score: {self.test_avg_score}\n")
        return ret

    def __str__(self):
        """Return a string representation of the object."""
        return repr(self)
