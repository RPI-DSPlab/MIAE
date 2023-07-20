from base import AbstractGeneralDataset
from torch.utils.data import DataLoader


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
