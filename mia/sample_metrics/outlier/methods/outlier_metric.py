from sample_metrics.base import ExampleMetric


class OutlierDetectionMetric(ExampleMetric):
    """
    A class used to represent an Outlier Detection Metric.

    This class inherits from the ExampleMetric class and implements the `compute_metric` method
    which is intended to compute the outlier detection metric on the provided data.
    """

    def __init__(self, config):
        """
        Initialize the OutlierDetectionMetric instance by invoking the initialization method of the superclass.

        Args:
            config (dict): A dictionary containing the configuration parameters for OneClassSVM.
        """
        super().__init__()
        self.config = config
        self.validate_config()

    def fit(self):
        """
        Fit the model given the data in config

        Raises:
            NotImplementedError: This method needs to be implemented by any class that inherits from
                                 OutlierDetectionMetric.
        """
        raise NotImplementedError("The 'fit' method must be implemented by the subclass.")

    def compute_metric(self):
        """
        Compute the outlier detection metric on the provided data. This method should be overridden by subclasses.

        Raises:
            NotImplementedError: This method needs to be implemented by any class that inherits from
                                 OutlierDetectionMetric.
        """
        raise NotImplementedError("The 'compute_metric' method must be implemented by the subclass.")

    def validate_config(self):
        """
        Check the validity of config files

        Raises:
            NotImplementedError: This method needs to be implemented by any class that inherits from
                                 OutlierDetectionMetric.
        """
        raise NotImplementedError("The 'validate_config' method must be implemented by the subclass.")
