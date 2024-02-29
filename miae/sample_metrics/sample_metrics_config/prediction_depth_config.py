from miae.sample_metrics.sample_metrics_config.base import ExampleHardnessConfig
from abc import ABC


class PredictionDepthConfig(ExampleHardnessConfig, ABC):
    """
    Configuration parameters for the PredictionDepthMetric.
    """

    """first we initialize the default configuration"""

    def __init__(self, config):
        # training parameters
        super().__init__(config)

        # pd only parameters
        self.knn_k = config.get("knn_k", 30)  # k nearest neighbors of knn classifier
