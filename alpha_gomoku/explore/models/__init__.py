from .cnn import *
from .unet import *
from .vit import *
from .value import *


class BoardToTensor(nn.Module):

    def forward(self, inputs):
        x, pls = inputs
        mask = (pls != 0)
        if mask.any():
            x[mask] = 1 - x[mask]
            x[x < 0] = 2
        return F.one_hot(x, 3).permute(0, 3, 1, 2).float().to(x.device)