from abc import ABC, abstractmethod

class ExampleHardnessConfig(ABC):
    """
    Base class for configuration parameters of the ExampleHardnessMetric.
    """
    @abstractmethod
    def __init__(self, config=None):
        """
        Initialize the ExampleHardnessConfig instance by providing a default configuration dictionary and updating it
        with the provided config dictionary.
        :param config: A dictionary containing the configuration parameters for ExampleHardnessMetric.
        """
        raise NotImplementedError("The '__init__' method must be implemented by the subclass.")


    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass