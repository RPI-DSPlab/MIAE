from utils.datasets.cifar10 import CIFAR10Dataset


def load_dataset(config):
    """Loads the datasets."""

    implemented_datasets = ('cifar10') # TODO: add more here
    assert config['dataset_name'] in implemented_datasets

    dataset = None

    if config['dataset_name'] == 'cifar10':
        dataset = CIFAR10Dataset(root_directory=config['data_path'],
                                 transform=config['transform'],
                                 target_transform=config['target_transform'])

    return dataset
