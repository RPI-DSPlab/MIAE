import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x, k=-1, train=True):
        """
        :param x: input
        :param k: output fms from the kth conv2d or the last layer
                k=-1: return the logits
        :param train: whether the model is in training mode
        """
        pass

    @abstractmethod
    def get_num_layers(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass