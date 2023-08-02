import pickle
from abc import ABC, abstractmethod


class ExampleMetric(ABC):
    def __init__(self):
        """
        Initialize the OutlierDetectionMetric instance by invoking the initialization method of the superclass.

        Args:
            config (dict): A dictionary containing the configuration parameters for OneClassSVM.
        """


    @abstractmethod
    def get_score(self, idx: int, train: bool = True):
        """
        get the score of the metric for the sample with index idx
        :param idx: index of the sample
        :param train: whether the sample is from the training set
        :return: the score of the metric
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def load_metric(self, path):
        """
        load the metric from the path
        :param path: path to load the metric
        """
        pass

    def validate_config(self):
        """
        Check the validity of config files

        Raises:
            NotImplementedError: This method needs to be implemented by any class that inherits from
                                 OutlierDetectionMetric.
        """
        raise NotImplementedError("The 'validate_config' method must be implemented by the subclass.")
