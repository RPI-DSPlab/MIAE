from miae.sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC


class IterationLearnedConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the IterationLearnedMetric.
    """

    def __init__(self, config):
        """
        Initialize the configuration parameters for the IterationLearnedMetric.
        :param config:
        """
        super().__init__(config)

        # il only parameters: either 'iteration_learned' or 'epoch_learned'
        self.learned_metric = config.get("learned_metric", "epoch_learned")


