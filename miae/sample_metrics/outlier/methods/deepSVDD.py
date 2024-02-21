import logging
import random

import numpy as np
import pandas as pd
import torch

from sample_metrics.outlier.methods.DeepSVDD_utils.deepSVDDCore import DeepSVDDNet
from sample_metrics.outlier.methods.outlier_metric import ExampleMetric, OutlierDetectionMetric
from utils.datasets.preprocessing import find_target_label_indices, apply_global_contrast_normalization
from utils.datasets.loader import load_dataset
from torch.utils.data import Subset

import torchvision.transforms as transforms


class DeepSVDD(OutlierDetectionMetric):
    """
    A class used to represent an Outlier Detection Metric using deep SVDD.

    This class inherits from the ExampleMetric class and implements the `compute_metric` method
    which is intended to compute the outlier detection metric on the provided data.
    """

    def __init__(self, config):
        """
        Initialize the OutlierDetectionMetric instance by invoking the initialization method of the superclass.
        Initialize the deep SVDD with the specified parameters.

        Args:
            config (dict): A dictionary containing the configuration parameters for deep SVDD.
        """
        super().__init__(config)
        self.deep_SVDD = None

        self.num_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([self.config['normal_class']])
        # TODO: 10 is a magic number and it is just for CIFAR 10, it needs fix
        self.outlier_classes = [i for i in range(10) if i != self.config['normal_class']]
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        if self.config['dataset_name'] == 'cifar10':
            # Pre-computed min and max values (after applying GCN) from training data per class
            class_specific_min_max = [(-28.94083453598571, 13.802961825439636),
                                      (-6.681770233365245, 9.158067708230273),
                                      (-34.924463588638204, 14.419298165027628),
                                      (-10.599172931391799, 11.093187820377565),
                                      (-11.945022995801637, 10.628045447867583),
                                      (-9.691969487694928, 8.948326776180823),
                                      (-9.174940012342555, 13.847014686472365),
                                      (-6.876682005899029, 12.282371383343161),
                                      (-15.603507135507172, 15.2464923804279),
                                      (-6.132882973622672, 8.046098172351265)]

            # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
            preprocessing_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: apply_global_contrast_normalization(x, normalization_scale='l1')),
                transforms.Normalize([class_specific_min_max[self.config['normal_class']][0]] * 3,
                                     [class_specific_min_max[self.config['normal_class']][1] -
                                      class_specific_min_max[self.config['normal_class']][0]] * 3)
            ])

            self.config['transform'] = preprocessing_transform
            self.config['target_transform'] = target_transform

        self.dataset = load_dataset(self.config)

        # Subset training set to normal class
        normal_class_indices = find_target_label_indices(self.dataset.train_set.targets, self.normal_classes)
        self.dataset.train_set = Subset(self.dataset.train_set, normal_class_indices)

    def calculate_label_score(self, data):
        """
        Calculate labels and scores for given data.

        Parameters:
        data (Tuple[torch.Tensor]): Tuple of inputs, labels and indices from the DataLoader.
            inputs (torch.Tensor): Input data.
            labels (torch.Tensor): Ground truth labels.
            idx (torch.Tensor): Indices of the data.
        net (BaseNet): The neural network model to use for prediction.

        Returns:
        List[Tuple[int, int, float]]: List of tuples with indices, labels and calculated scores.
        """
        inputs, labels, idx = data
        inputs = inputs.to(self.config['device'])
        outputs = self.deep_SVDD.net(inputs)
        dist = torch.sum((outputs - torch.tensor(self.deep_SVDD.c).to(self.config['device'])) ** 2, dim=1)

        if self.config['objective'] == 'soft-boundary':
            scores = dist - self.deep_SVDD.R ** 2
        else:
            scores = dist

        # Save triples of (idx, label, score) in a list
        return idx.cpu().data.numpy().tolist(), \
            labels.cpu().data.numpy().tolist(), \
            scores.cpu().data.numpy().tolist()

    def fit(self):
        """
        Fit the model using data as training input.
        """

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.config['xp_path'] + '/log.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Print arguments
        logger.info('\n---Program Start---')
        logger.info('Log file is %s.' % log_file)
        logger.info('Data path is %s.' % self.config['data_path'])
        logger.info('Export path is %s.' % self.config['xp_path'])
        logger.info("GPU is available." if torch.cuda.is_available() else "GPU is not available.")

        logger.info('Dataset: %s' % self.config['dataset_name'])
        # this is the # of class the current analyze is going on (0-9)
        logger.info('Normal class: %d' % self.config['normal_class'])
        logger.info('Network: %s' % self.config['net_name'])

        # Set seed
        if self.config['seed'] != -1:
            random.seed(self.config['seed'])
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
            logger.info('Set seed to %d.' % self.config['seed'])

        # Default device to 'cpu' if cuda is not available
        self.config['device'] = 'cpu' if not torch.cuda.is_available() else self.config['device']
        logger.info('Computation device: %s' % self.config['device'])
        logger.info('Number of dataloader workers: %d' % self.config['n_jobs_dataloader'])

        # Initialize DeepSVDD_utils model and set neural network \phi
        self.deep_SVDD = DeepSVDDNet(self.config['objective'], self.config['nu'])
        self.deep_SVDD.set_network(self.config['net_name'])

        # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
        if self.config['load_model']:
            self.deep_SVDD.load_model(model_path=self.config['load_model'], load_ae=True)
            logger.info('Loading model from %s.' % self.config['load_model'])

        logger.info('Pretraining: %s' % self.config['pretrain'])
        if self.config['pretrain']:
            # Log pretraining details
            logger.info('\n---Pretraining Start---')
            logger.info('Pretraining optimizer: %s' % self.config['ae_optimizer_name'])
            logger.info('Pretraining learning rate: %g' % self.config['ae_lr'])
            logger.info('Pretraining epochs: %d' % self.config['ae_n_epochs'])
            logger.info('Pretraining learning rate scheduler milestones: %s' % (self.config['ae_lr_milestone'],))
            logger.info('Pretraining batch size: %d' % self.config['ae_batch_size'])
            logger.info('Pretraining weight decay: %g' % self.config['ae_weight_decay'])

            # Pretrain model on datasets (via autoencoder)
            self.deep_SVDD.pretrain(self.dataset,
                                    optimizer_name=self.config['ae_optimizer_name'],
                                    lr=self.config['ae_lr'],
                                    n_epochs=self.config['ae_n_epochs'],
                                    lr_milestones=self.config['ae_lr_milestone'],
                                    batch_size=self.config['ae_batch_size'],
                                    weight_decay=self.config['ae_weight_decay'],
                                    device=self.config['device'],
                                    n_jobs_dataloader=self.config['n_jobs_dataloader'])

        # Log training details
        logger.info('\n---Training Start---')
        logger.info('Training optimizer: %s' % self.config['optimizer_name'])
        logger.info('Training learning rate: %g' % self.config['lr'])
        logger.info('Training epochs: %d' % self.config['n_epochs'])
        logger.info('Training learning rate scheduler milestones: %s' % (self.config['lr_milestone'],))
        logger.info('Training batch size: %d' % self.config['batch_size'])
        logger.info('Training weight decay: %g' % self.config['weight_decay'])

        # Train model on datasets
        self.deep_SVDD.train(self.dataset,
                             optimizer_name=self.config['optimizer_name'],
                             lr=self.config['lr'],
                             n_epochs=self.config['n_epochs'],
                             lr_milestones=self.config['lr_milestone'],
                             batch_size=self.config['batch_size'],
                             weight_decay=self.config['weight_decay'],
                             device=self.config['device'],
                             n_jobs_dataloader=self.config['n_jobs_dataloader'])

        # Save results, model, and self.configuration
        self.deep_SVDD.save_results(export_json=self.config['xp_path'] + '/results.json')
        self.deep_SVDD.save_model(export_model=self.config['xp_path'] + '/model.tar')

    def compute_metric(self):
        """
        Compute the outlier detection metric on the provided data.

        Returns:
            np.array: Predicted values for the data.
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.config['save_path'] + f"/log_class{self.config['normal_class']}.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Get train data loader
        train_loader, test_loader = self.dataset.loaders(batch_size=1, num_workers=8, shuffle_train=False,
                                                         shuffle_test=False)

        logger.info('Start evaluation...')

        # Calculate scores for the entire CIFAR10 dataset
        all_scores = []
        with torch.no_grad():
            for data in train_loader:
                index, label, score = self.calculate_label_score(data)
                # create a dictionary for each data instance
                # value = data[0].reshape(data[0].shape[0], -1)
                data_dict = {'Index': index[0], 'Label': label[0], 'Score': score[0]}
                all_scores.append(data_dict)

            for data in test_loader:
                index, label, score = self.calculate_label_score(data)
                # create a dictionary for each data instance
                # value = data[0].reshape(data[0].shape[0], -1)
                data_dict = {'Index': index[0], 'Label': label[0], 'Score': score[0]}
                all_scores.append(data_dict)

        # Save scores to a CSV file
        df_scores = pd.DataFrame(all_scores)
        df_scores.to_csv(self.config['save_path'] + f"/class_{self.config['normal_class']}_outlier_scores.csv",
                         index=False)
        logger.info("Anomaly scores saved to %s." % (self.config['save_path'] +
                                                     f"/class_{self.config['normal_class']}_outlier_scores.csv"))

    def validate_config(self):
        required_keys = ["dataset_name", "net_name", "load_model", "xp_path", "data_path", "objective", "nu",
                         "device",
                         "seed", "optimizer_name", "lr", "n_epochs", "lr_milestone", "batch_size",
                         "weight_decay",
                         "pretrain", "ae_optimizer_name", "ae_lr", "ae_n_epochs", "ae_lr_milestone",
                         "ae_batch_size",
                         "ae_weight_decay", "n_jobs_dataloader", "normal_class", "save_path"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing key in config: {key}")

        # TODO: Further validation could be done here, such as checking the types of the values,
        # TODO: or checking if the values fall within expected ranges.

        return True


if __name__ == "__main__":
    sample_config = {
        "dataset_name": "cifar10",
        "net_name": "cifar10_LeNet",
        "load_model": False,
        "xp_path": "./",
        "data_path": "./data",
        "objective": "one-class",
        "nu": 0.1,
        "device": "cuda",
        "seed": 90,
        "optimizer_name": "adam",
        "lr": 0.001,
        "n_epochs": 2,
        "lr_milestone": [50],
        "batch_size": 128,
        "weight_decay": 1e-6,
        "pretrain": True,
        "ae_optimizer_name": "adam",
        "ae_lr": 0.001,
        "ae_n_epochs": 1,
        "ae_lr_milestone": [0],
        "ae_batch_size": 128,
        "ae_weight_decay": 1e-6,
        "n_jobs_dataloader": 0,
        "normal_class": 0,
        "save_path": "./"
    }
    deepSvdd = DeepSVDD(sample_config)
    deepSvdd.fit()
    deepSvdd.compute_metric()
