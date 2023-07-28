from sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC


class PredictionDepthConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the PredictionDepthMetric.
    """

    """first we initialize the default configuration"""

    # training parameters
    num_epochs = 100  # number of epochs to train the model
    crit = 'cross_entropy'
    optimizer = 'sgd'
    num_iterations = 15000  # number of iterations to train the model
    seeds = [9203, 9304, 3456, 5210]  # seed values
    batch_size = 10000  # batch size
    lr = 0.04  # learning rate
    # result parameters
    save_path = 'il_results'  # directory to save results
    save_result = True  # save results
    knn_k = 30  # k nearest neighbors of knn classifier
