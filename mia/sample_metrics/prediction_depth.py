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
from sample_metrics.sample_metrics_config.prediction_depth_config import PredictionDepthConfig
from sample_metrics import sm_util


class PdHardness(ExampleMetric, ABC):
    """Compute the hardness of a dataset based on the prediction depth metric"""

    def __init__(self, config: PredictionDepthConfig, model: smmodels, dataset: smdatasets):
        """
        Initialize the PredictionDepthMetric instance by providing a configuration object, a model, and a dataset.
        :param config: A PredictionDepthConfig object
        :param model: A model object, which should be an instance of utils.models
        :param dataset: A dataset object, which should be an instance of utils.datasets
        """
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.trainloader = None  # for training
        self.testloader = None  # for testing during training
        self.trainloader2 = None  # support dataset for training depth's KNN prediction
        self.testloader2 = None  # support dataset for prediction depth's KNN prediction
        self.ready = False  # whether the metric is ready to be used (True means it's trained or loaded from a checkpoint)

        self.save_path = config.save_path  # path to save the model and results
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.modelckps = os.path.join(self.save_path, "modelckps")  # path to save the model checkpoints
        if not os.path.exists(self.modelckps):
            os.makedirs(self.modelckps)
        self.result_path = os.path.join(self.save_path, "results")  # path to save the results
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.result_avg_path = os.path.join(self.save_path, "results_avg")  # path to save the averaged results
        if not os.path.exists(self.result_avg_path):
            os.makedirs(self.result_avg_path)

        self.model = model
        self.trainloader, self.testloader = dataset.loaders(batch_size=config.batch_size, shuffle_train=True,
                                                            shuffle_test=False)
        self.trainloader2, self.testloader2 = dataset.loaders(batch_size=config.batch_size, shuffle_train=True,
                                                              shuffle_test=False)

        if config.crit == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Criterion {} not implemented".format(config.crit))

        if config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9,
                                             weight_decay=5e-4)
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(config.optimizer))


    def _knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t, rm_top1=True, dist='l2'):
        """
        knn prediction
        :param feature: feature vector of the current evaluating batch (dim = [B, F]
        :param feature_bank: feature bank of the support set (dim = [F, K]
        :param feature_labels: labels of the support set (dim = [K]
        :param classes: number of classes
        :param knn_k: number of nearest neighbors
        :param knn_t: temperature
        :param rm_top1: whether to remove the nearest pt of current evaluating pt in the train split (explain: this is because
                        the feature vector of the current evaluating pt may also be in the feature bank)
        :param dist: distance metric
        :return: prediction scores for each class (dim = [B, classes]
        """
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        feature_bank = feature_bank.t()  # [F, K].t() -> [K, F]
        B, F = feature.shape  # dim of feature vector of the current evaluating pt
        K, F = feature_bank.shape  # dim feature bank
        """
        B: batch size (ie: 200)
        F: feature dimension (ie: 65536)
        K: number of pts in the feature bank (ie 5000)
        """

        if dist == 'l2':
            knn_dist = 2

        distances = torch.cdist(feature, feature_bank, p=knn_dist)

        # Find the k nearest neighbors of the input feature.
        nearest_neighbors = distances.argsort(dim=1)[:, :knn_k]

        # If `rm_top1` is True, remove the nearest neighbor of the current evaluating point from the list of nearest neighbors.
        if rm_top1:
            mask = torch.ones(nearest_neighbors.shape[1], dtype=torch.bool)
            mask[0] = False  # mask the first element
            nearest_neighbors_dropped = nearest_neighbors[:, mask]
            nearest_labels = feature_labels[nearest_neighbors_dropped]
        else:
            nearest_labels = feature_labels[nearest_neighbors]

        # Compute the weighted scores using the inverse distances
        inv_distances = (1.0 / distances[:, :knn_k])
        knn_scores = torch.zeros(B, classes, device=feature.device)

        for i in range(B):
            for j in range(knn_k - 1 if rm_top1 else knn_k):
                knn_scores[i, nearest_labels[i, j]] += inv_distances[i, j]

        # Apply temperature scaling
        knn_scores /= knn_t

        return knn_scores

    def _get_feature_bank_from_kth_layer(self, model, dataloader, k, config):
        """
        Get feature bank from kth layer of the model
        :param model: the model
        :param dataloader: the dataloader
        :param k: the kth layer
        :param config: the configuration object
        :return: the feature bank (k-th layer feature for each datapoint) and
                the all label bank (ground truth label for each datapoint)
        """
        # NOTE: dataloader now has the return format of '(img, target), index'
        with torch.no_grad():
            for (img, all_label), idx in dataloader:
                img = img.cuda(non_blocking=True)  # an image from the dataset
                all_label = all_label.cuda(non_blocking=True)

                # the return of model():'None, _fm.view(_fm.shape[0], -1)  # B x (C x F x F)'
                _, fms = model(img, k, train=False)

        return fms, all_label  # somehow, the shape of fms is (number of image) * (it's feature map size)

    def _get_knn_prds_k_layer(self, model: smmodels, evaloader, floader, k, config: PredictionDepthConfig, rm_top=True):
        """
        Get the knn predictions for the kth layer
        :param model: the model
        :param evaloader: the evaluation dataloader (training or validation)
        :param floader: the feature dataloader (support set)
        :param k: the kth layer
        """
        knn_labels_all = []
        knn_conf_gt_all = []  # This statistics can be noisy
        indices_all = []

        f_bank, all_labels = self._get_feature_bank_from_kth_layer(model, floader,
                                                                   k,
                                                                   config)  # get the feature bank and all labels for the support set
        f_bank = f_bank.t().contiguous()
        with torch.no_grad():
            for j, ((imgs, labels), idx) in enumerate(evaloader):
                imgs = imgs.cuda(non_blocking=True)
                labels_b = labels.cuda(non_blocking=True)
                nm_cls = model.get_num_classes()
                _, inp_f_curr = model(imgs, k, train=False)

                knn_scores = self._knn_predict(inp_f_curr, f_bank, all_labels, classes=nm_cls, knn_k=config.knn_k,
                                               knn_t=1,
                                               rm_top1=rm_top)  # B x C
                knn_probs = F.normalize(knn_scores, p=1, dim=1)
                knn_labels_prd = knn_probs.argmax(1)
                knn_conf_gt = knn_probs.gather(dim=1, index=labels_b[:, None])  # B x 1
                knn_labels_all.append(knn_labels_prd)
                knn_conf_gt_all.append(knn_conf_gt)
                indices_all.append(idx)
            knn_labels_all = torch.cat(knn_labels_all, dim=0)  # N x 1
            knn_conf_gt_all = torch.cat(knn_conf_gt_all, dim=0).squeeze()
            indices_all = np.concatenate(indices_all, 0)
        return knn_labels_all, knn_conf_gt_all, indices_all

    def _get_prediction_depth(self, knn_labels_all, max_prediction_depth):
        """
        get prediction depth for a sample. reverse knn labels list and increase the counter until the label is different
        :param knn_labels_all:
        :return:
        """
        pd = 0
        knn_labels_all = list(reversed(knn_labels_all))
        while knn_labels_all[pd] == knn_labels_all[0] and pd <= max_prediction_depth - 2:
            pd += 1
        return max_prediction_depth - pd

    def _trainer(self, model):
        trainloader = self.trainloader
        testloader = self.testloader
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device
        config = self.config
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
            cos_scheduler.step()
            train_acc /= len(trainloader.dataset)
            train_loss /= len(trainloader.dataset)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            end_time_train = time.time()
            with torch.no_grad():

                test_acc = 0
                test_loss = 0
                for imgs, labels, idx in testloader:
                    imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    test_acc += (predicted == labels).sum().item()
                    test_loss += loss.item()
                test_acc /= len(testloader.dataset)
                test_loss /= len(testloader.dataset)
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc)

            if curr_iteration > config.num_iterations:
                break
            end_time_after_inference = time.time()
            if epoch % 20 == 0:
                print(
                    'Epoch: {}, Training Loss: {:.6f}, Test Loss: {:.6f}, Training Accuracy: {:.2f}, Test Accuracy: {:.2f}, training time: {:.2f}, inference time: '
                    '{:.6f}'.format(epoch, train_loss, test_loss, train_acc, test_acc,
                                    end_time_train - start_time_train,
                                    end_time_after_inference - end_time_train))

        return model, history

    def train_metric(self):
        """NOTE that this function call is optional, you could also load a trained model from the checkpoint"""
        seeds = self.config.seeds
        model_copy = copy.deepcopy(self.model)  # we need to copy the model because we will train it multiple times
        for seed in seeds:
            sm_util.set_seed(seed)
            model_copy = model_copy.to(self.device)
            # train the model
            model_copy, history = self._trainer(model_copy)
            # save the model
            torch.save(model_copy.state_dict(), os.path.join(self.modelckps, "ms{}_{}sgd{}"
                                                             .format(repr(model_copy), repr(self.dataset), seed)))

            index_knn_y_train = collections.defaultdict(list)
            index_pd_train = collections.defaultdict(int)
            knn_gt_conf_all_train = collections.defaultdict(list)
            index_knn_y_test = collections.defaultdict(list)
            index_pd_test = collections.defaultdict(int)
            knn_gt_conf_all_test = collections.defaultdict(list)

            # ------------------ training set pd ------------------
            if not os.path.exists(os.path.join(self.result_path)):
                os.makedirs(os.path.join(self.result_path))
            print("----- start obtaining training set pd -----")
            for k in tqdm(range(model_copy.get_num_layers())):
                knn_labels, knn_conf_gt_all, indices_all = self._get_knn_prds_k_layer(model_copy, self.trainloader,
                                                                                      self.trainloader2,
                                                                                      k, self.config, True)
                for idx, knn_l, knn_conf_gt in zip(indices_all, knn_labels, knn_conf_gt_all):
                    index_knn_y_train[int(idx)].append(knn_l.item())
                    knn_gt_conf_all_train[int(idx)].append(knn_conf_gt.item())
            for idx, knn_ls in index_knn_y_train.items():
                index_pd_train[idx] = (self._get_prediction_depth(knn_ls, model_copy.get_num_layers()))
            with open(os.path.join(self.result_path, 'pds{}train_seed{}_{}_trainpd.json'.format(repr(model_copy), seed,
                                                                                                repr(self.dataset))),
                      'w') as f:
                json.dump(index_pd_train, f)

            # ------------------ testing set pd ------------------
            print("----- start obtaining testing set pd -----")
            for k in tqdm(range(model_copy.get_num_layers())):
                knn_labels, knn_conf_gt_all, indices_all = self._get_knn_prds_k_layer(model_copy, self.testloader,
                                                                                      self.testloader2,
                                                                                      k, self.config, True)
                for idx, knn_l, knn_conf_gt in zip(indices_all, knn_labels, knn_conf_gt_all):
                    index_knn_y_test[int(idx)].append(knn_l.item())
                    knn_gt_conf_all_test[int(idx)].append(knn_conf_gt.item())
            for idx, knn_ls in index_knn_y_test.items():
                index_pd_test[idx] = (self._get_prediction_depth(knn_ls, model_copy.get_num_layers()))
            with open(os.path.join(self.result_path, 'pds{}train_seed{}_{}_testpd.json'.format(repr(model_copy), seed,
                                                                                               repr(self.dataset))),
                      'w') as f:
                json.dump(index_pd_test, f)

        # average the results
        self.train_avg_score = sm_util.avg_result(self.result_path, file_suf="_trainpd.json", roundToInt=False)
        self.test_avg_score = sm_util.avg_result(self.result_path, file_suf="_testpd.json", roundToInt=False)
        with open(os.path.join(self.result_avg_path,
                               'pds{}train_{}_trainpd.json'.format(repr(model_copy), repr(self.dataset))),
                  'w') as f:
            json.dump(self.train_avg_score, f)
        with open(os.path.join(self.result_avg_path,
                               'pds{}train_{}_testpd.json'.format(repr(model_copy), repr(self.dataset))),
                  'w') as f:
            json.dump(self.test_avg_score, f)
        self.ready = True

    def load_metric(self, path: str):
        """Load the trained metric from a checkpoint"""
        try:
            self.train_avg_score = sm_util.avg_result(path, file_suf="_trainpd.json", roundToInt=False)
            self.test_avg_score = sm_util.avg_result(path, file_suf="_testpd.json", roundToInt=False)
        except:
            raise ValueError(
                "train_avg_score or test_avg_score not found in the checkpoint, please train the metric first")
        self.ready = True

    def get_score(self, idx: int, train: bool):
        """
        Get the prediction depth score for a data sample
        :param idx: the index of the data sample
        :param train: whether the data sample is from the training set
        :return: the prediction depth score
        """
        if not self.ready:
            raise ValueError("Metric not ready, please train or load the metric first")
        if train:
            if idx not in self.train_avg_score:
                raise ValueError("Index {} not found in the training set".format(idx))
            return self.train_avg_score[idx]
        else:
            if idx not in self.test_avg_score:
                raise ValueError("Index {} not found in the testing set".format(idx))
            return self.test_avg_score[idx]

    def __repr__(self):
        ret = (f"Prediction depth metric:\n"
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
        return self.__repr__()
