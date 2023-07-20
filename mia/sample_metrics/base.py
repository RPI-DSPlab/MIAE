from abc import ABC, abstractmethod


class ExampleMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_score(self, idx: int, train: bool):
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
    def load(self, path):
        """
        load the metric from the path
        :param path: path to load the metric
        """
        pass
