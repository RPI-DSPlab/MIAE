from sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC


class ConsistencyScoreConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the ConsistencyScoreMetric.
    """

    """first we initialize the default configuration"""

    def __init__(self):
        super().__init__()
        self.learned_metric = 'consistency_score'  # either 'iteration_learned' or 'epoch_learned'
        # training parameters
        self.num_epochs = 100  # number of epochs to train the model
        self.crit = 'cross_entropy'
        self.optimizer = 'sgd'
        self.num_iterations = 15000  # number of iterations to train the model
        self.seed = 9203
        self.batch_size = 10000  # batch size
        self.lr = 0.04  # learning rate
        # result parameters
        self.save_path = 'cs_results'  # directory to save results
        self.save_result = True  # save results
        # consistency score parameters
        self.ss_ratio = 0.9  # ratio of subset set ie: 0.9 means 90% of the data are in-data and 10% are out-data
        self.n_runs = 200  # number of runs
