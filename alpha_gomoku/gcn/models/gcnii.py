import math

import torch
import torch.nn as nn

from .base import NetworkBase
from .base import get_laplacian_matrix


class GraphConvolutionII(nn.Module):

    def __init__(self, in_dim, dim, residual=False, variant=False, radius=6):
        super(GraphConvolutionII, self).__init__() 
        self.register_buffer(
            'laplacian_matrix', get_laplacian_matrix(radius)
        )
        self.variant = variant
        if self.variant:
            self.in_dim = 2 * in_dim 
        else:
            self.in_dim = in_dim
        self.dim = dim
        self.residual = residual
        self.weight = nn.Parameter(torch.FloatTensor(self.in_dim, self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.dim)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, h0, lmd, alpha, l):
        theta = math.log(lmd / l + 1)
        hi = torch.matmul(self.laplacian_matrix, x)
        if self.variant:
            support = torch.cat([hi, h0], -1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        out = (1 - theta) * r + theta * torch.matmul(support, self.weight)
        if self.residual:
            out = out + x
        return out


class GCNII(NetworkBase):
    
    def __init__(
            self, in_dim, hidden_dim, layer_num, 
            dropout=0.5, lmd=0.5, alpha=0.5, 
            residual=True, variant=False, radius=6
        ):
        super(GCNII, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, hidden_dim), nn.ReLU()
        )
        convs = nn.ModuleList()
        for _ in range(layer_num):
            convs.append(GraphConvolutionII(
                hidden_dim, hidden_dim, residual=residual, 
                variant=variant, radius=radius
            ))
        self.convs = nn.ModuleList(convs)
        self.classfier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.act_func = nn.ReLU()
        self.lmd = lmd
        self.alpha = alpha
        
    def forward(self, x):
        dropout = self.dropout
        x = self.preprocess(x)
        out = x
        for i, conv in enumerate(self.convs, 1):
            out = self.act_func(conv(dropout(out), x, self.lmd, self.alpha, i))
        return self.classfier(dropout(out)).squeeze(-1)
        