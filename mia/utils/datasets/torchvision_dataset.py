from base import AbstractGeneralDataset
from torch.utils.data import DataLoader, Subset


class PredefinedTorchvisionDataset(AbstractGeneralDataset):
    """
    Class for handling datasets predefined in torchvision.datasets.
    Extends the BaseADDataset class.
    """

    def __init__(self, root_directory: str):
        """
        Initializes the PredefinedTorchvisionDataset class.

        Args:
            root_directory (str): The root directory where the datasets is stored or will be downloaded to.
        """
        super().__init__(root_directory)

    def loaders(self, batch_size: int, shuffle_train: bool = True, shuffle_test: bool = False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        """
        Generates DataLoader instances for the training and testing sets.

        Args:
            batch_size (int): The number of samples per batch to load.
            shuffle_train (bool, optional): Whether to shuffle the training datasets. Defaults to True.
            shuffle_test (bool, optional): Whether to shuffle the testing datasets. Defaults to False.
            num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 0.

        Returns:
            tuple: A tuple containing the DataLoader instances for the training and testing sets respectively.
        """
        training_data_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                          num_workers=num_workers)
        testing_data_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                         num_workers=num_workers)

        return training_data_loader, testing_data_loader

    def subset_loaders(self,
                batch_size: int,
                train_indices: list,
                test_indices: list,
                shuffle_training_data=True,
                shuffle_testing_data=False,
                number_of_workers: int = 0) -> (DataLoader, DataLoader):
        """
        Generates DataLoader instances for the training and testing sets.
        :param batch_size: The number of samples per batch.
        :param train_indices: indices of the training set's subset
        :param test_indices: indices of the testing set's subset
        :param shuffle_training_data: Whether to shuffle the training data. Defaults to True.
        :param shuffle_testing_data: Whether to shuffle the testing data. Defaults to False.
        :param number_of_workers: The number of worker processes to use for data loading. Defaults to 0.
        :return: DataLoader, DataLoader: A tuple containing the DataLoader instances for the training and testing sets.
        """
        if len(train_indices) != 0:
            subset_train = Subset(self.train_set, train_indices)
            trainloader = DataLoader(subset_train, batch_size=batch_size, shuffle=shuffle_training_data,
                                     num_workers=number_of_workers)
        else:
            trainloader = None

        if len(test_indices) != 0:
            subset_test = Subset(self.test_set, test_indices)
            testloader = DataLoader(subset_test, batch_size=batch_size, shuffle=shuffle_testing_data,
                                    num_workers=number_of_workers)
        else:
            testloader = None
        return trainloader, testloader

