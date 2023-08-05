from utils.datasets.cifar10 import CIFAR10Dataset


def load_dataset(dataset_name, data_path, train_transform, test_transform, target_transform):
    """Loads the datasets."""

    implemented_datasets = ('cifar10')  # TODO: add more here
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'cifar10':
        dataset = CIFAR10Dataset(root_directory=data_path,
                                 train_transform=train_transform,
                                 test_transform=test_transform,
                                 target_transform=target_transform)

    return dataset
