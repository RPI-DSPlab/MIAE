from abc import ABC, abstractmethod


class BaseVisualization(ABC):
    @abstractmethod
    def __init__(self, config):
        """
        Initialize the OutlierDetectionMetric instance by invoking the initialization method of the superclass.

        Args:
            config (dict): A dictionary containing the configuration parameters for OneClassSVM.
        """
        self.config=config

    @abstractmethod
    def get_plot(self):
        """
        get the plot given by implementation
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass
