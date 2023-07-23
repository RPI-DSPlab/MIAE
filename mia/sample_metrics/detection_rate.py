from sample_metrics.base import ExampleMetric


class DetectionRate(ExampleMetric):
    def __init__(self):
        pass

    def get_score(self, idx: int, train: bool):
        """
        get the score of the metric for the sample with index idx
        :param idx: index of the sample
        :param train: whether the sample is from the training set
        :return: the score of the metric
        """
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def load(self, path):
        """
        load the metric from the path
        :param path: path to load the metric
        """
        pass
