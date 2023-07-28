from abc import ABC, abstractmethod


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

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
