import inspect
import json
from abc import ABC, abstractmethod
from datetime import datetime


class ExampleHardnessConfig(ABC):
    """
    Base class for configuration parameters of the ExampleHardnessMetric.
    In its child class, the default configuration parameters are defined as class attributes.
    """

    def __init__(self, config=None):
        """
        Initialize the ExampleHardnessConfig instance by providing a default configuration dictionary and updating it
        with the provided config dictionary.
        :param config: A dictionary containing the configuration parameters for ExampleHardnessMetric.
        """
        # training configurations
        self.num_epochs = config.get("num_epochs", 100)
        self.crit = config.get("crit", "cross_entropy")
        self.optimizer = config.get("optimizer", "sgd")
        self.seeds = config.get("seeds", [0, 1, 2, 3])
        self.batch_size = config.get("batch_size", 512)
        self.lr = config.get("lr", 0.1)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 0.0001)

        # saving/logging configurations
        self.save_result = config.get("save_result", True)
        self.save_path = config.get("save_path", "il_results")
        self.log_path = config.get("log_path", None)

    def update(self, config):
        """
        Update the configuration with the provided dictionary.

        :param config: A dictionary containing the configuration parameters to update.
        """

    def __repr__(self):
        """
        Return a string representation of the class attributes.
        """
        ret_builder = self.__class__.__name__ + ":\n"
        for i in inspect.getmembers(self):

            # to remove private and protected functions
            if not i[0].startswith('_'):

                # To remove other methods that does not start with an underscore
                if not inspect.ismethod(i[1]):
                    ret_builder += f"{i[0]} = {i[1]}\n"
        return str(ret_builder)

    def __str__(self):
        """
        Return the type of the configuration.
        """
        return self.__class__.__name__

    def save(self, path, name=None):
        """
        Save the configuration to a json file.

        :param path: The path to the file to save the configuration to.
        :param name: The name of the file to save the configuration to. If None, the name of the file is the name of the
        class concatenated with the current date and time.
        """
        if name is None:
            name = self.__class__.__name__ + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        with open(path + name, "w") as f:
            json.dump(self.__dict__, f)

    def load(self, path):
        """
        Load the configuration from a json file.

        :param path: The path to the file to load the configuration from.
        """
        with open(path, "r") as f:
            config = json.load(f)
        self.update(config)
