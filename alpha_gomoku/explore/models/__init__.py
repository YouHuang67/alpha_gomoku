from .cnn import *


class BoardToTensor(nn.Module):

    def forward(self, x):
        return F.one_hot(x, 3).permute(0, 3, 1, 2).float().to(x.device)