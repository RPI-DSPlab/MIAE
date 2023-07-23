from torch import nn


class ConvNet(nn.Module):
    def __init__(self, scales=3, filters=32, pooling='max', init_weights=True):
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

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc(x)
        return x

