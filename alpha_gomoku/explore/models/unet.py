import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv(nn.Module):

    def __init__(self, input_dim, output_dim, stride, padding,
                 kernel_size=3, shortcut=False):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
        )
        if shortcut and input_dim == output_dim and stride == 1:
            self.conv_skip = nn.Sequential()
        else:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
                nn.BatchNorm2d(output_dim),
            )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):

    def __init__(self, channel, widen_factor=8):
        super(ResUnet, self).__init__()
        filters = [
            8 * widen_factor, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Conv2d(filters[0], 1, 1, 1, bias=True)

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        return self.output_layer(x10)


class KernelOneResNet(nn.Module):

    def __init__(self, channel, depth, output_dim=64):
        super(KernelOneResNet, self).__init__()
        assert depth % 2 == 0

        pre_conv = ResidualConv(
            channel, output_dim, 1, 0, 1, shortcut=True
        )
        self.backbone = nn.Sequential(pre_conv, *[
            ResidualConv(output_dim, output_dim, 1, 0, 1, shortcut=True)
            for _ in range(depth // 2 - 1)
        ])

    def forward(self, x):
        return self.backbone(x)


class KernelOneResUnet(nn.Module):

    def __init__(self, depth, widen_factor=8):
        super(KernelOneResUnet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            KernelOneResNet(16, depth, 8 * widen_factor)
        )
        self.core = ResUnet(8 * widen_factor, 2)

    def forward(self, x):
        out = self.feature(x)
        size = 2 ** int(np.ceil(np.log(out.size(-1)) / np.log(2)))
        out = F.interpolate(out, size, mode='bilinear')
        out = self.core(out)
        return F.interpolate(out, x.size(-1), mode='bilinear').view(x.size(0), -1)


def KernelOneUNet20_8():
    return KernelOneResUnet(20, 8)


def KernelOneUNet40_2():
    return KernelOneResUnet(40, 2)


def KernelOneUNet40_4():
    return KernelOneResUnet(40, 4)


def KernelOneUNet40_8():
    return KernelOneResUnet(40, 8)


def KernelOneUNet60_2():
    return KernelOneResUnet(60, 2)


def KernelOneUNet60_4():
    return KernelOneResUnet(60, 4)


def KernelOneUNet60_8():
    return KernelOneResUnet(60, 8)


def KernelOneUNet100_2():
    return KernelOneResUnet(100, 2)


def KernelOneUNet160_2():
    return KernelOneResUnet(160, 2)