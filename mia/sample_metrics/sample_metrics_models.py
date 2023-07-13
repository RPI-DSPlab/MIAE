"""This python file contains the models used for computing the sample metrics."""

import torch.nn as nn
import torch
from torchvision.models import vgg16

class VGG16(nn.Module):
    """
    VGG model for computing Prediction Depth
    """
    def __init__(
            self,
            encoder=None,
            num_classes=100
    ):
        """
        initializer of VGG16 model for computing Prediction Depth
        :param encoder: the encoder of the VGG model, should be passed when a instance of VGG16 is created
        :param num_classes: number of classes
        """
        super(VGG16, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, x, k=0, train=True):
        """

        :param x:
        :param k: output fms from the kth conv2d or the last layer
        :return:
        """
        n_layer = 0
        _fm = None
        for m in self.encoder.children():  # x goes through all the layers
            x = m(x)
            if not train:
                if isinstance(m, nn.Conv2d):
                    if n_layer == k: # returns here if we are getting the feature map from this layer
                        return None, x.view(x.shape[0], -1) # B x (C x F x F)
                    n_layer += 1
        logits = self.classifier(x)
        if not train:
            if k == n_layer:
                _fm = torch.softmax(logits, 1)
                return None, _fm.view(_fm.shape[0], -1)  # B x (C x F x F)
        else:
            return logits

    def get_num_layers(self):
        """
        :return: number of layers in the encoder
        """
        return 14

def getvgg16(num_class=10):
    """
    Get the VGG16 model
    :param num_class: number of classes
    """
    ecd = vgg16().features
    model = VGG16(ecd, num_class)
    return model