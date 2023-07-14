from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class AbstractAnomalyDetectionDataset(ABC):
    """
    Base class for anomaly detection datasets.
    This class defines the basic structure for datasets to be used in anomaly detection tasks.
    """

    def __init__(self, root_directory: str):
        """
        Initialize the AbstractAnomalyDetectionDataset instance.

        Args:
            root_directory (str): The root directory path where the datasets is located.
        """
        super().__init__()
        self.root_directory = root_directory  # root path to data

        self.number_of_classes = 2  # 0: normal, 1: outlier
        self.normal_class_labels = None  # tuple with original class labels that define the normal class
        self.outlier_class_labels = None  # tuple with original class labels that define the outlier class

        self.training_set = None  # must be of type torch.utils.data.Dataset
        self.testing_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def create_data_loaders(self,
                            batch_size: int,
                            shuffle_training_data=True,
                            shuffle_testing_data=False,
                            number_of_workers: int = 0) -> (DataLoader, DataLoader):
        """
        Abstract method to implement data loaders of type torch.utils.data.DataLoader for the training and testing sets.

        Args:
            batch_size (int): The number of samples per batch.
            shuffle_training_data (bool, optional): Whether to shuffle the training data. Defaults to True.
            shuffle_testing_data (bool, optional): Whether to shuffle the testing data. Defaults to False.
            number_of_workers (int, optional): The number of worker processes to use for data loading. Defaults to 0.

        Returns:
            DataLoader, DataLoader: A tuple containing the DataLoader instances for the training and testing sets.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.
        """
        pass

    def __repr__(self) -> str:
        """
        String representation of the AbstractAnomalyDetectionDataset instance.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__
