from utils.models.conv_net import ConvNet
from utils.models.wide_resnet import WideResNet


def init_network(arch: str, init_weights=True):
    if arch == 'cnn32-3-max':
        return ConvNet(scales=3, filters=32, pooling='max', init_weights=init_weights)
    elif arch == 'cnn32-3-mean':
        return ConvNet(scales=3, filters=32, pooling='mean', init_weights=init_weights)
    elif arch == 'cnn64-3-max':
        return ConvNet(scales=3, filters=64, pooling='max', init_weights=init_weights)
    elif arch == 'cnn64-3-mean':
        return ConvNet(scales=3, filters=64, pooling='mean', init_weights=init_weights)
    elif arch == 'wrn28-1':
        return WideResNet(depth=28, num_classes=10, widen_factor=1, init_weights=init_weights)
    elif arch == 'wrn28-2':
        return WideResNet(depth=28, num_classes=10, widen_factor=2, init_weights=init_weights)
    elif arch == 'wrn28-10':
        return WideResNet(depth=28, num_classes=10, widen_factor=10, init_weights=init_weights)
    else:
        raise ValueError('Architecture not recognized', arch)