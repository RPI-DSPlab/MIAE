from torch import nn

from utils.models.base import BaseModel


class WideResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate, widen_factor):
        super(WideResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(nn.functional.relu(self.bn1(x))))
        out = self.conv2(nn.functional.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(BaseModel):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0, init_weights=True):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(WideResidualBlock, nStages[1], n, stride=1,
                                       dropout_rate=dropout_rate, widen_factor=k)
        self.layer2 = self._wide_layer(WideResidualBlock, nStages[2], n, stride=2,
                                       dropout_rate=dropout_rate, widen_factor=k)
        self.layer3 = self._wide_layer(WideResidualBlock, nStages[3], n, stride=2,
                                       dropout_rate=dropout_rate, widen_factor=k)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

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

    def _wide_layer(self, block, planes, num_blocks, stride, dropout_rate, widen_factor):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate, widen_factor))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, k=-1, train=True):  # TODO: implement k function if necessary
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.functional.relu(self.bn1(out))
        out = nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def get_num_layers(self):
        return sum(1 for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))

    def __repr__(self):
        return f'WideResNet(depth={self.depth}, num_classes={self.num_classes}, widen_factor={self.widen_factor}, ' \
               f'dropout_rate={self.dropout_rate})'

