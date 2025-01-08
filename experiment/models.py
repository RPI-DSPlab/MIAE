# some architectures are adapted from the repo
# https://github.com/DennisLiu2022/Membership-Inference-Attacks-by-Exploiting-Loss-Trajectory/blob/main/architectures.py
import math

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def get_model(model_name, num_classes, input_size):
    if model_name == 'resnet56':
        return create_resnet56(num_classes=num_classes, input_size=input_size)
    elif model_name == 'wrn32_4':
        return create_wideresnet32_4(num_classes=num_classes, input_size=input_size)
    elif model_name == 'wrn28_2':
        return create_wideresnet28_2(num_classes=num_classes, input_size=input_size)
    elif model_name == 'vgg16':
        return create_vgg16(num_classes=num_classes, input_size=input_size)
    elif model_name == 'mobilenet':
        return create_mobilenet(num_classes=num_classes, input_size=input_size)
    elif model_name == 'mlp_for_texas_purchase':
        return create_mlp(input_size=input_size, num_classes=num_classes, layer_sizes=[512, 256, 128, 64])
    else:
        raise ValueError('Unknown model name')
    


def create_mlp(input_size, num_classes, layer_sizes):
    return MLP(input_size=input_size, num_classes=num_classes, layer_sizes=layer_sizes)

def create_resnet56(num_classes=10, input_size=32):
    num_blocks = [9, 9, 9]
    num_classes = num_classes
    input_size = input_size
    block_type = 'basic'
    return ResNet(num_blocks=num_blocks, num_classes=num_classes, input_size=input_size, block_type=block_type)


def create_wideresnet32_4(num_classes=10, input_size=32):
    num_blocks = [5, 5, 5]
    widen_factor = 4
    dropout_rate = 0.3
    return WideResNet(num_blocks=num_blocks, widen_factor=widen_factor, num_classes=num_classes,
                      dropout_rate=dropout_rate, input_size=input_size)

def create_wideresnet28_2(num_classes=10, input_size=32):
    num_blocks = [4, 4, 4]
    widen_factor = 2
    dropout_rate = 0.3
    return WideResNet(num_blocks=num_blocks, widen_factor=widen_factor, num_classes=num_classes,
                      dropout_rate=dropout_rate, input_size=input_size)


def create_vgg16(num_classes=10, input_size=32):
    conv_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    fc_layers = [512, 512]
    max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    conv_batch_norm = True
    init_weights = True
    return VGG(input_size=input_size, num_classes=num_classes, conv_channels=conv_channels, fc_layers=fc_layers,
               max_pool_sizes=max_pool_sizes, conv_batch_norm=conv_batch_norm, init_weights=init_weights)


def create_mobilenet(num_classes=10, input_size=32):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    return MobileNet(cfg=cfg, num_classes=num_classes, input_size=input_size)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()

        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU(inplace=True))

        self.layers = layers

    def forward(self, x):
        fwd = self.layers[0](x)
        fwd += self.layers[1](x)
        fwd = self.layers[2](fwd)
        return fwd


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes, input_size, block_type='basic'):
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.input_size = input_size
        self.block_type = block_type
        self.in_channels = 16
        self.num_output = 1

        if self.block_type == 'basic':
            self.block = BasicBlock

        init_conv = []

        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))

        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(Flatten())
        end_layers.append(nn.Linear(64 * self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers

    def forward(self, x, k=None, train=True):
        """

        :param x:
        :param k: output fms from the kth conv2d or the last layer
        :return:
        """
        if k is None:
            out = self.init_conv(x)

            for layer in self.layers:
                out = layer(out)

            out = self.end_layers(out)

            return out

        # the following is for getting feature maps
        out = self.init_conv(x)

        n_layer = 0
        _fm = None

        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if not train:
                if isinstance(layer, BasicBlock):
                    if n_layer == k:
                        return None, out.view(out.size(0), -1)
                    n_layer += 1

        out = self.end_layers(out)
        if not train:
            if k == n_layer:
                _fm = torch.softmax(out, 1)
                return None, _fm.view(_fm.size(0), -1)
        else:
            return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class wide_basic(nn.Module):
    def __init__(self, in_channels, channels, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.layers = nn.ModuleList()
        conv_layer = []
        conv_layer.append(nn.BatchNorm2d(in_channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True))
        conv_layer.append(nn.Dropout(p=dropout_rate))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=True))

        self.layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=True),
            )

        self.layers.append(shortcut)

    def forward(self, x):
        out = self.layers[0](x)
        out += self.layers[1](x)
        return out

class WideResNet(nn.Module):
    def __init__(self, num_blocks, widen_factor, num_classes, dropout_rate, input_size):
        super(WideResNet, self).__init__()
        self.num_blocks = num_blocks
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.in_channels = 16
        self.num_output = 1

        self.init_conv = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.layers = nn.ModuleList()
        self.layers.extend(self._wide_layer(wide_basic, self.in_channels * self.widen_factor, block_id=0, stride=1))
        self.layers.extend(self._wide_layer(wide_basic, 32 * self.widen_factor, block_id=1, stride=2))
        self.layers.extend(self._wide_layer(wide_basic, 64 * self.widen_factor, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.BatchNorm2d(64 * self.widen_factor, momentum=0.9))
        end_layers.append(nn.ReLU(inplace=True))
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(Flatten())
        end_layers.append(nn.Linear(64 * self.widen_factor, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

    def _wide_layer(self, block, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, self.dropout_rate, stride))
            self.in_channels = channels
        return layers

    def forward(self, x, k=0, train=True):
        """

        :param x:
        :param k: output fms from the kth conv2d or the last layer
        :return:
        """
        if k is None:
            out = self.init_conv(x)

            for layer in self.layers:
                out = layer(out)

            out = self.end_layers(out)

            return out

        # the following is for getting feature maps
        out = self.init_conv(x)

        n_layer = 0
        _fm = None

        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if not train:
                if isinstance(layer, wide_basic):
                    if n_layer == k:
                        return None, out.view(out.size(0), -1)
                    n_layer += 1

        out = self.end_layers(out)
        if not train:
            if k == n_layer:
                _fm = torch.softmax(out, 1)
                return None, _fm.view(_fm.size(0), -1)
        else:
            return out


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_features(self, x):
        out = self.init_conv(x)

        for layer in self.layers:
            out = layer(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class VGG(nn.Module):
    def __init__(self, input_size, num_classes, conv_channels, fc_layers, max_pool_sizes, conv_batch_norm,
                 init_weights):
        super(VGG, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.fc_layer_sizes = fc_layers
        self.max_pool_sizes = max_pool_sizes
        self.conv_batch_norm = conv_batch_norm
        self.init_weights = init_weights

        self.num_output = 1

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel

        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x, k=None, train=True):
        """

        :param x:
        :param k: output fms from the kth conv2d or the last layer
        :return:
        """
        n_layer = 0
        _fm = None

        fwd = self.init_conv(x)

        if k is None:  # regular forward
            for layer in self.layers:
                fwd = layer(fwd)

            fwd = self.end_layers(fwd)
            return fwd

        else:  # get feature map
            for layer in self.layers:
                fwd = layer(fwd)
                if not train:
                    if n_layer == k:  # returns here if we are getting the feature map from this layer
                        return fwd.view(fwd.shape[0], -1)  # B x (C x F x F)
                    n_layer += 1

            fwd = self.end_layers(fwd)
            if not train:
                if k == n_layer:
                    _fm = torch.softmax(fwd, 1)
                    return _fm.view(_fm.shape[0], -1)  # B x (C x F x F)
            else:
                return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def get_num_layers(self):
            return 14

    def get_features(self, x):
        out = self.init_conv(x)

        for layer in self.layers:
            out = layer(out)

        return out

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class MobileNet(nn.Module):
    def __init__(self, cfg, num_classes, input_size):
        super(MobileNet, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_output = 1
        self.in_channels = 32
        init_conv = []

        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []

        end_layers.append(nn.AvgPool2d(2))

        end_layers.append(Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return layers

    def forward(self, x):
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def get_features(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        return fwd
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class WideResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate, widen_factor):
        super(WideResidualBlock, self).__init__()
        assert widen_factor > 0, "Widen factor must be greater than 0"
        mid_planes = out_planes * widen_factor

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, layer_sizes[i]))
            else:
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        self.initialize_weights()

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)