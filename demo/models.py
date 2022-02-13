import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cppboard import Board
from .utils import WEIGHT_DIR


class BoardToTensor(nn.Module):

    def forward(self, inputs):
        x, pls = inputs
        mask = (pls != 0)
        if mask.any():
            x[mask] = 1 - x[mask]
            x[x < 0] = 2
        return F.one_hot(x, 3).permute(0, 3, 1, 2).float().to(x.device)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride,
                 drop_rate=0.0, reduction=16):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
            nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False) or None
        self.se_layer = SELayer(out_planes, reduction)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes,
                 stride, drop_rate=0.0, reduction=16):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            block = BasicBlock(i == 0 and in_planes or out_planes, out_planes,
                               i == 0 and stride or 1, drop_rate, reduction)
            layers.append(block)
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor=1, drop_rate=0.0, reduction=16):
        assert((depth - 4) % 6 == 0)
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, *nChannels[0:2], 1, drop_rate, reduction)
        self.block2 = NetworkBlock(n, *nChannels[1:3], 1, drop_rate, reduction)
        self.block3 = NetworkBlock(n, *nChannels[2:4], 1, drop_rate, reduction)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Conv2d(nChannels[3], 1, 1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return self.fc(out).view(x.size(0), -1)


class EnsembleOutput(nn.Module):

    def __init__(self):
        super(EnsembleOutput, self).__init__()
        self.bn = nn.BatchNorm1d(Board.BOARD_SIZE ** 2)
        self.fc = nn.Linear(Board.BOARD_SIZE ** 2, 1, bias=True)

    def forward(self, x):
        out = self.bn(x.view(x.size(0), -1))
        out = self.fc(out).view(-1)
        out = torch.tanh(out)
        return x, out


def SEWideResNet16_1(drop_rate=0.0, reduction=1):
    return nn.Sequential(BoardToTensor(),
                         WideResNet(16, 1, drop_rate, reduction),
                         EnsembleOutput())


def SEWideResNet16_2(drop_rate=0.0, reduction=1):
    return nn.Sequential(BoardToTensor(),
                         WideResNet(16, 2, drop_rate, reduction),
                         EnsembleOutput())


def get(model_name, **kwargs):
    func = {k.lower(): v for k, v in globals().items()}[model_name.lower()]
    args = inspect.getfullargspec(func).args
    model = func(**{arg: kwargs[arg] for arg in set(args) & set(kwargs)})
    weight_path = WEIGHT_DIR / f'{model_name.lower()}.pth'
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    return model
