import time

from base import ExampleMetric
import sys
import torch
import os
import numpy as np
import json
import sample_metric_dataset as smdataset
import sample_metrics_models as smmodels

class PdHardness(ExampleMetric):
    """Compute the hardness of a dataset based on the prediction depth metric"""
    def __init__(self, config: dict):
        """Initialize the metric with the configuration"""
        self.config = config
        self.trainloader = None  # for training
        self.testloader = None  # for testing during training
        self.trainloader2 = None  # support dataset for training depth's KNN prediction
        self.testloader2 = None  # support dataset for prediction depth's KNN prediction

        if "model" not in config:
            raise ValueError("Model not specified")
        if "dataset" not in config:
            raise ValueError("Dataset not specified")
        if "crit" not in config:
            raise ValueError("Criterion not specified")
        if "optimizer" not in config:
            raise ValueError("Optimizer not specified")
        if "lr" not in config:
            raise ValueError("Learning rate not specified")

        if config["model"] == 'vgg16':
            if config["dataset"] == 'cifar10':
                self.model = smmodels.getvgg16(num_classes=10)
            elif config["dataset"] == 'cifar100':
                self.model = smmodels.getvgg16(num_classes=100)
            else:
                raise NotImplementedError("Dataset {} not implemented".format(config.dataset))
        else:
            raise NotImplementedError("Model {} not implemented".format(config.model))

        if config["crit"] == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Criterion {} not implemented".format(config.crit))

        if config["optimizer"] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"], momentum=0.9,
                                             weight_decay=5e-4)
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(config.optimizer))

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def load_data(self, dataset=None):
        self.dataset = None if dataset is not None else self.config['dataset']
        try:
            self.trainloader, self.testloader, self.trainloader2, self.testloader2 = smdataset.load_data(self.config)
        except NotImplementedError as error:
            print(str(error), "dataset specified from config is :{}".format(dataset), file=sys.stderr)

    def _knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t, rm_top1=True, dist='l2'):
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

    def _get_feature_bank_from_kth_layer(model, dataloader, k, args):
        """
        Get feature bank from kth layer of the model
        :param model: the model
        :param dataloader: the dataloader
        :param k: the kth layer
        :return: the feature bank (k-th layer feature for each datapoint) and
                the all label bank (ground truth label for each datapoint)
        """
        # NOTE: dataloader now has the return format of '(img, target), index'
        with torch.no_grad():
            for (img, all_label), idx in dataloader:
                img = img.cuda(non_blocking=True)  # an image from the dataset
                all_label = all_label.cuda(non_blocking=True)

                # the return of model():'None, _fm.view(_fm.shape[0], -1)  # B x (C x F x F)'
                if args.half:
                    with torch.autocast():
                        _, fms = model(img, k, train=False)
                else:
                    _, fms = model(img, k, train=False)

        return fms, all_label  # somehow, the shape of fms is (number of image) * (it's feature map size)

    def _get_knn_prds_k_layer(model, evaloader, floader, k, args, rm_top=True):
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

        f_bank, all_labels = PdHardness._get_feature_bank_from_kth_layer(model, floader,
                                                              k,
                                                              args)  # get the feature bank and all labels for the support set
        f_bank = f_bank.t().contiguous()
        with torch.no_grad():
            for j, ((imgs, labels), idx) in enumerate(evaloader):
                imgs = imgs.cuda(non_blocking=True)
                labels_b = labels.cuda(non_blocking=True)
                nm_cls = args.num_classes
                if args.half:
                    with torch.autocast():
                        _, inp_f_curr = model(imgs, k, train=False)
                else:
                    _, inp_f_curr = model(imgs, k, train=False)
                """
                Explanation of the following function:
                knn_predict(inp_f_curr, f_bank, all_labels, classes=nm_cls, knn_k=args.knn_k, knn_t=1, rm_top1=train_split)
                inp_f_curr is the feature of the image (batch of images) we want to predict it's label
                f_bank is the feature bank of the support set, and we know its ground truth label given all_labels
                We want to use information from the support set (f_bank) to predict the label of the image (inp_f_curr)
                """
                knn_scores = PdHardness._knn_predict(inp_f_curr, f_bank, all_labels, classes=nm_cls, knn_k=args.knn_k, knn_t=1,
                                         rm_top1=rm_top)  # B x C
                knn_probs = torch.nn.functional.normalize(knn_scores, p=1, dim=1)
                knn_labels_prd = knn_probs.argmax(1)
                knn_conf_gt = knn_probs.gather(dim=1, index=labels_b[:, None])  # B x 1
                knn_labels_all.append(knn_labels_prd)
                knn_conf_gt_all.append(knn_conf_gt)
                indices_all.append(idx)
            knn_labels_all = torch.cat(knn_labels_all, dim=0)  # N x 1
            knn_conf_gt_all = torch.cat(knn_conf_gt_all, dim=0).squeeze()
            indices_all = np.concatenate(indices_all, 0)
        return knn_labels_all, knn_conf_gt_all, indices_all

    def _get_prediction_depth(knn_labels_all, max_prediction_depth):
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

    def _trainer(self):
        trainloader = self.trainloader
        testloader = self.testloader
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device
        args = self.args
        curr_iteration = 0
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        print('------ Training started on {} with total number of {} epochs ------'.format(device, args.num_epochs))
        for epoch in range(args.num_epochs):
            # time each epoch
            start_time_train = time.time()
            train_acc = 0
            train_loss = 0
            for (imgs, labels), idx in trainloader:
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
                for (imgs, labels), idx in testloader:
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

            if curr_iteration > args.iterations:
                break
            end_time_after_inference = time.time()
            if epoch % 20 == 0:
                print(
                    'Epoch: {}, Training Loss: {:.6f}, Test Loss: {:.6f}, Training Accuracy: {:.2f}, Test Accuracy: {:.2f}, training time: {:.2f}, inference time: '
                    '{:.6f}'.format(epoch, train_loss, test_loss, train_acc, test_acc,
                                    end_time_train - start_time_train,
                                    end_time_after_inference - end_time_train))

        return model, history

    def train_model(self):
        """NOTE that this function call is optional, you could also load a trained model from the checkpoint"""
    def compute_metric(self):
        pass



class IterHardness(ExampleMetric):
    def __init__(self):
        pass

    def compute_metric(self):
        pass
