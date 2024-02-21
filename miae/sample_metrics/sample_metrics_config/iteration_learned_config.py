from miae.sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC


class IterationLearnedConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the IterationLearnedMetric.
    """

    """first we initialize the default configuration"""
    def __init__(self):
        super().__init__()
        self.learned_metric = 'iteration_learned'  # either 'iteration_learned' or 'epoch_learned'
        # training parameters
        self.num_epochs = 100  # number of epochs to train the model
        self.crit = 'cross_entropy'
        self.optimizer = 'sgd'
        self.num_iterations = 15000  # number of iterations to train the model
        self.seeds = [9203, 9304, 3456, 5210]  # seed values
        self.batch_size = 10000  # batch size
        self.lr = 0.04  # learning rate
        # result parameters
        self.save_path = 'il_results'  # directory to save results
        self.save_result = True  # save results


