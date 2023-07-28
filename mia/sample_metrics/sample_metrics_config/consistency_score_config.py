from sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC


class ConsistencyScoreConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the ConsistencyScoreMetric.
    """

    """first we initialize the default configuration"""

    learned_metric = 'consistency_score'  # either 'iteration_learned' or 'epoch_learned'
    # training parameters
    num_epochs = 100  # number of epochs to train the model
    crit = 'cross_entropy'
    optimizer = 'sgd'
    num_iterations = 15000  # number of iterations to train the model
    seed = 9203
    batch_size = 10000  # batch size
    lr = 0.04  # learning rate
    # result parameters
    save_path = 'cs_results'  # directory to save results
    save_result = True  # save results
    # consistency score parameters
    ss_ratio = 0.9  # ratio of subset set ie: 0.9 means 90% of the data are in-data and 10% are out-data
    n_runs = 200  # number of runs

    def __init__(self, config=None):
        """
        Initialize the ConsistencyScoreConfig instance by providing a default configuration dictionary and updating it
        with the provided config dictionary.

        :param config: A dictionary containing the configuration parameters for ConsistencyScoreMetric.
        """

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
        String representation of the ConsistencyScoreConfig class.

        :return: String representation of the ConsistencyScoreConfig class.
        """

        return "ConsistencyScoreConfig({})".format(self.__dict__)

    def __str__(self):
        """
        String representation of the ConsistencyScoreConfig class.

        :return: String representation of the ConsistencyScoreConfig class.
        """

        return self.__repr__()
