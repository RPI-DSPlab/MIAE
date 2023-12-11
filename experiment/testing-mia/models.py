import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, scales=3, filters=32, pooling='max', init_weights=True, num_channel=3):
        super(ConvNet, self).__init__()

        layers = []
        for i in range(scales):
            if i == 0:
                layers.append(nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(filters, filters*2, kernel_size=3, stride=1, padding=1))
                filters *= 2
            layers.append(nn.ReLU(inplace=True))

            if pooling == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pooling == 'mean':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(filters*4*4, 10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # forward for debugging
    # def forward(self, x):
    #     for layer in self.conv:
    #         x = layer(x)
    #         print(f'After layer: {layer.__class__.__name__}, shape: {x.shape}')
    #     x = x.view(x.size(0), -1)  # flatten the tensor
    #     print(f'After flattening, shape: {x.shape}')
    #     x = self.fc(x)
    #     print(f'After FC, shape: {x.shape}')
    #     return x


    def forward(self, x, k=0, train=True):
        n_layer = 0
        for m in self.conv.children():

            x = m(x)
            if not train:
                if isinstance(m, nn.Conv2d):
                    if n_layer == k:
                        return None, x.view(x.shape[0], -1)  # B x (C x F x F)
                    n_layer += 1

        logits = self.fc(x.view(x.size(0), -1))
        if not train:
            if k == n_layer:
                _fm = F.relu(x)  # You can apply ReLU to the feature map before returning
                return None, _fm.view(_fm.shape[0], -1)  # B x (C x F x F)
        else:
            return logits

    def get_num_layers(self):
        """
        Get the total number of layers in the ConvNet model.
        """
        # Count the number of layers in the convolutional part (excluding the final fully connected layer)
        num_layers = len([layer for layer in self.conv if isinstance(layer, nn.Conv2d)])

        # Add 1 for the final fully connected layer
        num_layers += 1

        return num_layers

    def __str__(self):
        return f"cnn{self.filters}-{self.scales}-{self.pooling}"
