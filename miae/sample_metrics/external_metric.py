# The purpose of external metric is for case when we need to load a
# pre-trained metric from an external source. This is useful when
# that metric takes a long time to train, and we want to reuse it
import pickle
from miae.sample_metrics.base import ExampleMetric


class ExternalMetric(ExampleMetric):
    def __init__(self, score_arr):
        """
        External metric only have score array
        """
        self.score_arr = score_arr

    def get_score(self, idx: int):
        return self.score_arr[idx]

    def __str__(self):
        return str(self.metric)

    def __repr__(self):
        return repr(self.metric)

    def load_metric(self, path):
        with open(path, "rb") as f:
            self.metric = pickle.load(f)
