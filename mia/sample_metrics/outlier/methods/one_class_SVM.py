from sklearn import svm

from sample_metrics.outlier.methods.outlier_metric import OutlierDetectionMetric


class OneClassSVM(OutlierDetectionMetric):
    """
    A class used to represent an Outlier Detection Metric using one-class SVM.

    This class inherits from the ExampleMetric class and implements the `compute_metric` method
    which is intended to compute the outlier detection metric on the provided data.
    """

    def __init__(self, config):
        """
        Initialize the OutlierDetectionMetric instance by invoking the initialization method of the superclass.
        Initialize the OneClassSVM with the specified parameters.

        Args:
            config (dict): A dictionary containing the configuration parameters for OneClassSVM.
        """
        super().__init__()
        self.one_class_svm = svm.OneClassSVM(nu=config['nu'], kernel=config['kernel'], gamma=config['gamma'])

    def fit(self, data):
        """
        Fit the model using data as training input.

        Args:
            data: Training data.
        """
        self.one_class_svm.fit(data)

    def compute_metric(self, data):
        """
        Compute the outlier detection metric on the provided data.

        Args:
            data: The data on which to compute the outlier detection metric.

        Returns:
            np.array: Predicted values for the data.
        """
        return self.one_class_svm.predict(data)
