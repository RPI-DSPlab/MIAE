from PIL import Image
from torchvision.datasets import CIFAR10
from utils.datasets.torchvision_dataset import PredefinedTorchvisionDataset


class CIFAR10Dataset(PredefinedTorchvisionDataset):
    """
    A class that represents a custom CIFAR10 datasets.
    """

    def __init__(self, root_directory: str, train_transform, test_transform, target_transform):
        """
        Initialize the CustomCIFAR10Dataset.

        Args:
            root_directory (str): The root directory where the datasets is located or will be downloaded.
        """
        super().__init__(root_directory)

        # Create training set
        self.train_set = IndexedCIFAR10(root=root_directory, train=True, download=True,
                                        transform=train_transform, target_transform=target_transform)

        # Create test set
        self.test_set = IndexedCIFAR10(root=root_directory, train=False, download=True,
                                       transform=test_transform, target_transform=target_transform)


class IndexedCIFAR10(CIFAR10):
    """
    A CIFAR10 class that also returns the index of a data sample when accessed.
    """

    def __getitem__(self, index: int):
        """
        Override the method of the CIFAR10 class to also return the index of a data sample.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple (image, target, index), where 'image' is the input image, 'target' is the label of the
                   sample, and 'index' is the index of the sample in the datasets.
        """
        if self.train:
            image, target = self.data[index], self.targets[index]
        else:
            image, target = self.data[index], self.targets[index]

        # Convert to PIL Image for consistency with other datasets
        image = Image.fromarray(image)

        if self.transform is not None and self.train:
            image = self.transform(image)

        if self.target_transform is not None and not self.train:
            image = self.transform(image)

        return image, target, index
