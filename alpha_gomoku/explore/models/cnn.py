import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    expansion = 1
    kernel_size = 3
    bottleneck = False

    def __init__(self, in_planes, out_planes, stride=1):
        super(PreActBlock, self).__init__()
        expansion = self.expansion
        kernel_size = self.kernel_size
        if self.bottleneck:
            in_planes = [in_planes, out_planes, out_planes]
            out_planes = [out_planes, out_planes, expansion * out_planes]
            kernel_size = [1, kernel_size, 1]
            stride = [1, stride, 1]
        else:
            in_planes = [in_planes, expansion * out_planes]
            out_planes = [expansion * out_planes, expansion * out_planes]
            kernel_size = [kernel_size, kernel_size]
            stride = [stride, 1]
        self.bn1 = nn.BatchNorm2d(in_planes[0])
        self.conv1 = nn.Conv2d(
            in_planes[0], out_planes[0], kernel_size=kernel_size[0],
            stride=stride[0], padding=(kernel_size[0] - 1) // 2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes[1])
        self.conv2 = nn.Conv2d(
            in_planes[1], out_planes[1], kernel_size=kernel_size[1],
            stride=stride[1], padding=(kernel_size[1] - 1) // 2, bias=False
        )
        if self.bottleneck:
            self.bn3 = nn.BatchNorm2d(in_planes[2])
            self.conv3 = nn.Conv2d(
                in_planes[2], out_planes[2], kernel_size=kernel_size[2],
                stride=stride[2], padding=(kernel_size[2] - 1) // 2, bias=False
            )

        if stride[-2] != 1 or in_planes[0] != out_planes[-1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes[0], out_planes[-1], kernel_size=1,
                          stride=stride[-2], bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.bottleneck:
            out = self.conv3(F.relu(self.bn3(out)))
        return out + shortcut


class PreBottleneckBlock(PreActBlock):
    expansion = 4
    kernel_size = 3
    bottleneck = True


class KernelOneBlock(PreActBlock):
    expansion = 1
    kernel_size = 1
    bottleneck = False


class KernelOneBottleneckBlock(PreBottleneckBlock):
    expansion = 4
    kernel_size = 1
    bottleneck = True


class PreActResNet(nn.Module):

    def __init__(self, num_blocks, bottleneck, kernel_one_level):
        super(PreActResNet, self).__init__()
        if bottleneck:
            blocks = [KernelOneBottleneckBlock for _ in range(kernel_one_level)]
            blocks += [PreBottleneckBlock for _ in range(4 - len(blocks))]
        else:
            blocks = [KernelOneBlock for _ in range(kernel_one_level)]
            blocks += [PreActBlock for _ in range(4 - len(blocks))]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(blocks[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(blocks[2], 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(blocks[3], 512, num_blocks[3], stride=1)
        self.bn = nn.BatchNorm2d(512 * blocks[-1].expansion)
        self.linear = nn.Conv2d(
            512 * blocks[-1].expansion, 1, kernel_size=1, stride=1, bias=True
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = self.linear(out)
        return out.view(x.size(0), -1)


def PreActResNet18(kernel_one_level=0):
    return PreActResNet([2, 2, 2, 2], False, kernel_one_level=kernel_one_level)


def PreActResNet34(kernel_one_level=0):
    return PreActResNet([3, 4, 6, 3], False, kernel_one_level=kernel_one_level)


def PreActResNet50(kernel_one_level=0):
    return PreActResNet([3, 4, 6, 3], True, kernel_one_level=kernel_one_level)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=(kernel_size - 1) // 2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False
        )
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
        ) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
                 drop_rate=0.0, kernel_size=3):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
                                      nb_layers, stride, drop_rate, kernel_size)

    def _make_layer(self, block, in_planes, out_planes,
                    nb_layers, stride, drop_rate, kernel_size):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor=1, drop_rate=0.0, kernel_one_level=0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate,
                                   (1 if kernel_one_level > 0 else 3))
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 1, drop_rate,
                                   (1 if kernel_one_level > 1 else 3))
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, drop_rate,
                                   (1 if kernel_one_level > 2 else 3))
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Conv2d(nChannels[3], 1, kernel_size=1, bias=True)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return self.fc(out).view(x.size(0), -1)


def WideResnet16_1(drop_rate=0.0, kernel_one_level=0):
    return WideResNet(16, 1, drop_rate, kernel_one_level)


def WideResnet16_2(drop_rate=0.0, kernel_one_level=0):
    return WideResNet(16, 2, drop_rate, kernel_one_level)


def WideResnet40_1(drop_rate=0.0, kernel_one_level=0):
    return WideResNet(40, 1, drop_rate, kernel_one_level)


def WideResnet40_2(drop_rate=0.0, kernel_one_level=0):
    return WideResNet(40, 2, drop_rate, kernel_one_level)


def WideResnet40_8(drop_rate=0.0, kernel_one_level=0):
    return WideResNet(40, 8, drop_rate, kernel_one_level)


def WideResnet64_1(drop_rate=0.0, kernel_one_level=0):
    return WideResNet(64, 1, drop_rate, kernel_one_level)


def WideResnet64_2(drop_rate=0.0, kernel_one_level=0):
    return WideResNet(64, 2, drop_rate, kernel_one_level)


class KernelOneWideResNet(nn.Module):

    def __init__(self, pre_depth, depth, widen_factor=2, drop_rate=0.0):
        super(KernelOneWideResNet, self).__init__()
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.feature = NetworkBlock(
            pre_depth // 2, nChannels[0], nChannels[1], block, 1, drop_rate, 1
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[1], nChannels[1], block, 1, drop_rate, 3)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 1, drop_rate, 3)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, drop_rate, 3)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Conv2d(nChannels[3], 1, kernel_size=1, bias=True)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.feature(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return self.fc(out).view(x.size(0), -1)


def KernelOneWideResNet40_16_1(drop_rate=0.0):
    return KernelOneWideResNet(40, 16, 1, drop_rate=drop_rate)


def KernelOneWideResNet40_16_2(drop_rate=0.0):
    return KernelOneWideResNet(40, 16, 2, drop_rate=drop_rate)


def KernelOneWideResNet40_40_1(drop_rate=0.0):
    return KernelOneWideResNet(40, 40, 1, drop_rate=drop_rate)


def KernelOneWideResNet40_40_2(drop_rate=0.0):
    return KernelOneWideResNet(40, 40, 2, drop_rate=drop_rate)


def KernelOneWideResNet60_16_1(drop_rate=0.0):
    return KernelOneWideResNet(60, 16, 1, drop_rate=drop_rate)


def KernelOneWideResNet60_16_2(drop_rate=0.0):
    return KernelOneWideResNet(60, 16, 2, drop_rate=drop_rate)


def KernelOneWideResNet80_16_2(drop_rate=0.0):
    return KernelOneWideResNet(80, 16, 2, drop_rate=drop_rate)


def KernelOneWideResNet160_16_1(drop_rate=0.0):
    return KernelOneWideResNet(160, 16, 1, drop_rate=drop_rate)


def KernelOneWideResNet160_16_2(drop_rate=0.0):
    return KernelOneWideResNet(160, 16, 2, drop_rate=drop_rate)