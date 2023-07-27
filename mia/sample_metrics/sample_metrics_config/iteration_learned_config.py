from sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC


class IterationLearnedConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the IterationLearnedMetric.
    """

    """first we initialize the default configuration"""

    learned_metric = 'iteration_learned'  # either 'iteration_learned' or 'epoch_learned'
    # training parameters
    num_epochs = 100  # number of epochs to train the model
    crit = 'cross_entropy'
    optimizer = 'sgd'
    num_iterations = 15000  # number of iterations to train the model
    seeds = [9203, 9304, 3456, 5210]  # seed values
    batch_size = 10000  # batch size
    lr = 0.04  # learning rate

    def __init__(self, config=None):
        """
        Initialize the IterationLearnedConfig instance by providing a default configuration dictionary and updating it
        with the provided config dictionary.

        :param config: A dictionary containing the configuration parameters for IterationLearnedMetric.
        """

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

    def __repr__(self):
        """
        String representation of the IterationLearnedConfig class.

        :return: String representation of the IterationLearnedConfig class.
        """

        return "IterationLearnedConfig({})".format(self.__dict__)

    def __str__(self):
        """
        String representation of the IterationLearnedConfig class.

        :return: String representation of the IterationLearnedConfig class.
        """

        return self.__repr__()
