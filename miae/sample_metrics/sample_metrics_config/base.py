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
        if config is not None:
            self.update(config)

    def update(self, config):
        """
        Update the configuration with the provided dictionary.

        :param config: A dictionary containing the configuration parameters to update.
        """
        # Use the update method to overwrite the default values with the provided values
        self.__dict__.update(config)

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
