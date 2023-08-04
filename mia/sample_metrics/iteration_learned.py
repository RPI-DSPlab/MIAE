import collections
import copy
import time
from abc import ABC

import torch.nn.functional as F
from tqdm import tqdm

from sample_metrics.base import ExampleMetric
import sys
import torch
import os
import numpy as np
import json

from utils import models as smmodels
from utils import datasets as smdatasets
from sample_metrics.sample_metrics_config import IterationLearnedConfig
from sample_metrics import sm_util


class IlHardness(ExampleMetric, ABC):
    """Computer the hardness of a dtaset based on the iteration learned method."""

    def __init__(self, config: IterationLearnedConfig, model: smmodels, dataset: smdatasets):
        """
        :param config: the configuration file
        :param model: the model
        :param dataset: the dataset
        """
        super().__init__()
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = None  # the trainloader for the dataset
        self.trainloader_inf = None  # the second trainloader for inference
        self.testloader = None  # the testloader for the dataset
        self.testloader_inf = None  # the second testloader for inference
        self.ready = False  # whether the trainloader and testloader are ready
        self.trainset = None

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
        self.trainloader, self.testloader = self.dataset.loaders(
            batch_size=config.batch_size, shuffle_train=True, shuffle_test=True)
        self.trainloader2, self.testloader2 = self.dataset.loaders(
            batch_size=config.batch_size, shuffle_train=True, shuffle_test=True)

        if config.crit == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Criterion {} not implemented".format(config.crit))

        if config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9,
                                             weight_decay=5e-4)
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(config.optimizer))

    def _trainer(self, trainloader, trainloader_inf, testloader_inf, model, optimizer, criterion, device,
                 learning_history_train_dict, learning_history_test_dict, config: IterationLearnedConfig):
        curr_iteration = 0
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        print('------ Training started on {} with total number of {} epochs ------'.format(device, config.num_epochs))
        for epoch in range(config.num_epochs):
            # time each epoch
            start_time_train = time.time()
            train_acc = 0
            train_loss = 0
            for imgs, labels, idx in trainloader:
                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item()
                train_loss += loss.item()
                curr_iteration += 1
                if config.learned_metric == "iteration":  # if we are using iteration learned metric
                    model.eval()
                    with torch.no_grad():
                        for imgs_inf, labels_inf, idx_inf in trainloader_inf:
                            imgs_inf, labels_inf = imgs_inf.cuda(non_blocking=True), labels_inf.cuda(non_blocking=True)
                            outputs_inf = model(imgs_inf)
                            _, predicted_inf = torch.max(outputs_inf.data, 1)
                            for i in range(len(idx_inf)):
                                learning_history_train_dict[idx_inf[i].item()].append(
                                    labels_inf[i].item() == predicted_inf[i].item())
                        for imgs_inf, labels_inf, idx_inf in testloader_inf:
                            imgs_inf, labels_inf = imgs_inf.cuda(non_blocking=True), labels_inf.cuda(non_blocking=True)
                            outputs_inf = model(imgs_inf)
                            _, predicted_inf = torch.max(outputs_inf.data, 1)
                            for i in range(len(idx_inf)):
                                learning_history_test_dict[idx_inf[i].item()].append(
                                    labels_inf[i].item() == predicted_inf[i].item())
                    model.train()

            cos_scheduler.step()
            train_acc /= len(trainloader.dataset)
            train_loss /= len(trainloader.dataset)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            end_time_train = time.time()

            if config.learned_metric == "epoch":  # if we are using epoch learned metric
                model.eval()
                with torch.no_grad():
                    for imgs_inf, labels_inf, idx_inf in trainloader_inf:
                        imgs_inf, labels_inf = imgs_inf.cuda(non_blocking=True), labels_inf.cuda(non_blocking=True)
                        outputs_inf = model(imgs_inf)
                        _, predicted_inf = torch.max(outputs_inf.data, 1)
                        for i in range(len(idx_inf)):
                            learning_history_train_dict[idx_inf[i].item()].append(
                                labels_inf[i].item() == predicted_inf[i].item())
                    for imgs_inf, labels_inf, idx_inf in testloader_inf:
                        imgs_inf, labels_inf = imgs_inf.cuda(non_blocking=True), labels_inf.cuda(non_blocking=True)
                        outputs_inf = model(imgs_inf)
                        _, predicted_inf = torch.max(outputs_inf.data, 1)
                        for i in range(len(idx_inf)):
                            learning_history_test_dict[idx_inf[i].item()].append(
                                labels_inf[i].item() == predicted_inf[i].item())
                model.train()
            if curr_iteration > config.num_iterations:
                break
            end_time_after_inference = time.time()
            if epoch % 20 == 0:
                print(
                    'Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}, training time: {:.2f}, inference time: '
                    '{:.6f}'.format(epoch, train_loss, train_acc, end_time_train - start_time_train,
                                    end_time_after_inference - end_time_train))

        def _determine_learned_metric(learning_history_dict):
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
        model_copy = copy.deepcopy(self.model)  # we need to copy the model so that we can train it multiple times

        for seed in seeds:
            sm_util.set_seed(seed)
            model_copy = model_copy.to(self.device)

            learning_history_train_dict = {}
            for _, idx in self.trainloader_inf:
                for i in idx:
                    learning_history_train_dict[i.item()] = list()
            learning_history_test_dict = {}
            for _, idx in self.testloader_inf:
                for i in idx:
                    learning_history_test_dict[i.item()] = list()
            # train the model
            self._trainer(self.trainloader, self.trainloader_inf, self.testloader_inf, model_copy, self.optimizer,
                          self.criterion, self.device, learning_history_train_dict, learning_history_test_dict,
                          self.config)

            # save the partial results
            if self.config.learned_metric == "iteration":
                with open(os.path.join(self.result_file_path,
                                       "{}-{}-learned_metric_iteration_seed{}_train.json".format(repr(self.dataset),
                                                                                                 repr(self.model),
                                                                                                 seed), "w")) as f:
                    json.dump(learning_history_train_dict, f)
                with open(os.path.join(self.result_file_path,
                                       "{}-{}-learned_metric_iteration_seed{}_test.json".format(repr(self.dataset),
                                                                                                repr(self.model),
                                                                                                seed), "w")) as f:
                    json.dump(learning_history_test_dict, f)
            elif self.config.learned_metric == "epoch":
                with open(os.path.join(self.result_file_path,
                                       "{}-{}-learned_metric_epoch_seed{}_train.json".format(repr(self.dataset),
                                                                                             repr(self.model),
                                                                                             seed), "w")) as f:
                    json.dump(learning_history_train_dict, f)
                with open(os.path.join(self.result_file_path,
                                       "{}-{}-learned_metric_epoch_seed{}_test.json".format(repr(self.dataset),
                                                                                            repr(self.model),
                                                                                            seed), "w")) as f:
                    json.dump(learning_history_test_dict, f)
            else:
                raise NotImplementedError("Learned metric {} not implemented".format(self.config.learned_metric))

        # average the results
        try:
            self.train_avg_score = sm_util.avg_result(self.result_file_path, file_suf="train.json")
            self.test_avg_score = sm_util.avg_result(self.result_file_path, file_suf="test.json")
        except Exception as e:
            raise Exception("No files found in the directory, please train the model first")

        # save the averaged results
        if self.config.learned_metric == "iteration":
            with open(os.path.join(self.result_file_path_avg,
                                   "{}-{}-learned_metric_iteration_train.json".format(repr(self.dataset),
                                                                                      repr(self.model)), "w")) as f:
                json.dump(self.train_avg_score, f)
            with open(os.path.join(self.result_file_path_avg,
                                   "{}-{}-learned_metric_iteration_test.json".format(repr(self.dataset),
                                                                                     repr(self.model)), "w")) as f:
                json.dump(self.test_avg_score, f)
        self.ready = True

    def load_metric(self, path: str):
        """
        This function loads the learned metric from the saved files
        """
        try:
            self.train_avg_score = sm_util.avg_result(path, file_suf="train.json")
            self.test_avg_score = sm_util.avg_result(path, file_suf="test.json")
        except Exception as e:
            raise Exception("No files found in the directory, please train the model first")
        self.ready = True

    def get_score(self, idx: int, train: bool):
        """
        This function returns the learned metric of a datapoint
        :param idx: the index of the datapoint
        :param train: whether the datapoint is in the training set or not
        :return: the learned metric of the datapoint
        """
        if not self.ready:
            raise Exception("Please train the model first")
        if train:
            return self.train_avg_score[str(idx)]
        else:
            return self.test_avg_score[str(idx)]

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
