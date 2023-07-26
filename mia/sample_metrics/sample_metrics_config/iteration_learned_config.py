from sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC

class IterationLearnedConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the IterationLearnedMetric.
    """

    def __init__(self, config=None):
        """
        Initialize the IterationLearnedConfig instance by providing a default configuration dictionary and updating it
        with the provided config dictionary.

        :param config: A dictionary containing the configuration parameters for IterationLearnedMetric.
        """

        """first we initialize the default configuration"""

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

        """then we update the default configuration with the provided config dictionary"""
        if config is None:
            return

        for key in config:
            if key in self.__dict__:
                self.__dict__[key] = config[key]
            else:
                raise ValueError("Invalid key in config: {}".format(key))

