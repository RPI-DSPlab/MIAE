import torch
import logging
import random
import numpy as np

from deepSVDDCore import DeepSVDDNet
from utils.datasets.loader import load_dataset


def get_results(config):

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config['xp_path'] + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('\n---Program Start---')
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % config['data_path'])
    logger.info('Export path is %s.' % config['xp_path'])
    logger.info("GPU is available." if torch.cuda.is_available() else "GPU is not available.")

    logger.info('Dataset: %s' % config['dataset_name'])
    logger.info('Normal class: %d' % config['normal_class']) # this is the # of class the current analyze is going on (0-9)
    logger.info('Network: %s' % config['net_name'])

    # Set seed
    if config['seed'] != -1:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        logger.info('Set seed to %d.' % config['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % config['device'])
    logger.info('Number of dataloader workers: %d' % config['n_jobs_dataloader'])

    # Load data
    dataset = load_dataset(config['dataset_name'], config['data_path'], config['normal_class'])

    # Initialize DeepSVDD_utils model and set neural network \phi
    deep_SVDD = DeepSVDD(config['objective'], config['nu'])
    deep_SVDD.set_network(config['net_name'])

    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if config['load_model']:
        deep_SVDD.load_model(model_path=config['load_model'], load_ae=True)
        logger.info('Loading model from %s.' % config['load_model'])

    logger.info('Pretraining: %s' % config['pretrain'])
    if config['pretrain']:
        # Log pretraining details
        logger.info('\n---Pretraining Start---')
        logger.info('Pretraining optimizer: %s' % config['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % config['ae_lr'])
        logger.info('Pretraining epochs: %d' % config['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (config['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % config['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % config['ae_weight_decay'])

        # Pretrain model on datasets (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=config['ae_optimizer_name'],
                           lr=config['ae_lr'],
                           n_epochs=config['ae_n_epochs'],
                           lr_milestones=config['ae_lr_milestone'],
                           batch_size=config['ae_batch_size'],
                           weight_decay=config['ae_weight_decay'],
                           device=config['device'],
                           n_jobs_dataloader=config['n_jobs_dataloader'])

    # Log training details
    logger.info('\n---Training Start---')
    logger.info('Training optimizer: %s' % config['optimizer_name'])
    logger.info('Training learning rate: %g' % config['lr'])
    logger.info('Training epochs: %d' % config['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (config['lr_milestone'],))
    logger.info('Training batch size: %d' % config['batch_size'])
    logger.info('Training weight decay: %g' % config['weight_decay'])

    # Train model on datasets
    deep_SVDD.train(dataset,
                    optimizer_name=config['optimizer_name'],
                    lr=config['lr'],
                    n_epochs=config['n_epochs'],
                    lr_milestones=config['lr_milestone'],
                    batch_size=config['batch_size'],
                    weight_decay=config['weight_decay'],
                    device=config['device'],
                    n_jobs_dataloader=config['n_jobs_dataloader'])

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=config['xp_path'] + '/results.json')
    deep_SVDD.save_model(export_model=config['xp_path'] + '/model.tar')
