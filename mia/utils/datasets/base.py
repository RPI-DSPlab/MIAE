from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class AbstractGeneralDataset(ABC):
    """
    Base class datasets for general purpose.
    This class defines the basic structure for datasets to be used in anomaly detection tasks.
    """

    def __init__(self, root_directory: str):
        """
        Initialize the Dataset instance.

        Args:
            root_directory (str): The root directory path where the datasets is located.
        """
        super().__init__()
        self.root_directory = root_directory  # root path to data

        self.number_of_classes = 2  # 0: normal, 1: outlier
        self.normal_class_labels = None  # tuple with original class labels that define the normal class
        self.outlier_class_labels = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self,
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

    @abstractmethod
    def subset_loaders(self,
                       batch_size: int,
                       train_indices: list,
                       test_indices: list,
                       shuffle_training_data=True,
                       shuffle_testing_data=False,
                       number_of_workers: int = 0) -> (DataLoader, DataLoader):
        """
        Abstract method to implement data loaders of type torch.utils.data.DataLoader for the training and testing sets.
        :param batch_size: The number of samples per batch.
        :param train_indices: indices of the training set's subset
        :param test_indices: indices of the testing set's subset
        :param shuffle_training_data: Whether to shuffle the training data. Defaults to True.
        :param shuffle_testing_data: Whether to shuffle the testing data. Defaults to False.
        :param number_of_workers: The number of worker processes to use for data loading. Defaults to 0.
        :return: DataLoader, DataLoader: A tuple containing the DataLoader instances for the training and testing sets.
        """
        pass

    def get_training_set(self):
        """
        Get the training set.

        Returns:
            torch.utils.data.Dataset: The training set.
        """
        return self.train_set

    def get_testing_set(self):
        """
        Get the testing set.

        Returns:
            torch.utils.data.Dataset: The testing set.
        """
        return self.test_set

    def get_num_classes(self) -> int:
        """
        Get the number of classes in the datasets.

        Returns:
            int: The number of classes in the datasets.
        """
        return self.number_of_classes

    def __repr__(self) -> str:
        """
        String representation of the AbstractAnomalyDetectionDataset instance.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__
