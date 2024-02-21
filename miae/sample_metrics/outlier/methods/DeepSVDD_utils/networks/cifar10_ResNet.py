import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_net import BaseNet


# Define the basic Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(residual)
        out = nn.ReLU()(out)
        return out


# Define the ResNet
class CIFAR10_ResNet(BaseNet):
    def __init__(self):
        super(CIFAR10_ResNet, self).__init__()
        self.in_channels = 64
        self.rep_dim = 512

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4 layers
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, self.rep_dim, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(self.rep_dim, self.rep_dim)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class CIFAR10_ResNet_Autoencoder(nn.Module):
    def __init__(self):
        super(CIFAR10_ResNet_Autoencoder, self).__init__()
        self.rep_dim = 512
        self.in_channels = 64

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4 layers
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, self.rep_dim, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(self.rep_dim, self.rep_dim)

        # Decoder
        self.fc_d = nn.Linear(self.rep_dim, self.rep_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        if debug: print(f"\n\nInput shape: {x.shape}")

        # Encoding
        out = self.conv(x)
        if debug: print(f"After first convolution: {out.shape}")
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        if debug: print(f"After layer1: {out.shape}")
        out = self.layer2(out)
        if debug: print(f"After layer2: {out.shape}")
        out = self.layer3(out)
        if debug: print(f"After layer3: {out.shape}")
        out = self.layer4(out)
        if debug: print(f"After layer4: {out.shape}")

        out = self.avg_pool(out)
        if debug: print(f"After avg_pool: {out.shape}")
        out = out.view(out.size(0), -1)
        if debug: print(f"After flattening: {out.shape}")
        out = self.fc(out)
        if debug: print(f"After FC: {out.shape}")

        x = F.relu(out)
        encoded = x.view(x.size(0), -1)  # flatten the output of the encoder
        # encoded = self.fc(x)
        # if debug: print(f"Encoded shape: {encoded.shape}")

        # Decoding
        x = F.relu(self.fc_d(encoded))
        if debug: print(f"After first decoder FC: {x.shape}")
        x = x.view(x.size(0), 512, 1, 1)  # reshape back into a 2D image
        if debug: print(f"After reshaping: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After first upsample: {x.shape}")

        x = F.relu(self.conv1(x))
        if debug: print(f"After first decoder conv: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After second upsample: {x.shape}")

        x = F.relu(self.conv2(x))
        if debug: print(f"After second decoder conv: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After third upsample: {x.shape}")

        x = F.relu(self.conv3(x))
        if debug: print(f"After third decoder conv: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After fourth upsample: {x.shape}")

        x = torch.sigmoid(self.conv4(x))
        if debug: print(f"Final output shape: {x.shape}")

        return x
