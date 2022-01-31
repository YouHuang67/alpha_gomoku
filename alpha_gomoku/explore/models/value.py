import torch.nn as nn
import torch.nn.functional as F


class ValueWrapper(nn.Module):

    def __init__(self, module, widen_factor=4):
        super(ValueWrapper, self).__init__()
        self.module = module
        dim = [m for m in module.modules()
               if isinstance(m, nn.Conv2d)][-1].in_channels
        self.project = nn.Sequential(
            nn.Linear(dim, widen_factor * dim, bias=True),
            nn.ReLU(),
            nn.Linear(widen_factor * dim, widen_factor * dim, bias=True),
            nn.ReLU(),
            nn.Linear(widen_factor * dim, 1, bias=True)
        )

    def forward(self, x):
        feat, prob_out = self.module(x, get_feat=True)
        value_out = self.project(
            F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
        ).view(-1)
        return prob_out, value_out
