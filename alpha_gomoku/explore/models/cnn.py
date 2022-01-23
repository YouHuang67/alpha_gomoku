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
