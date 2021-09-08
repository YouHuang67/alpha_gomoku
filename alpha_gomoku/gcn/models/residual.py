import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import utils
from ...cppboard import Board
from .base import initialize
from .base import NetworkBase
from .base import EmbeddingBase
from .base import get_laplacian_matrix


class PlayerEmbedding(EmbeddingBase):

    def __init__(self, dim):
        super(PlayerEmbedding, self).__init__()
        self.dim = dim
        self.register_buffer('embeddings', torch.randn(1, 3, dim))

    def forward(self, vectors):
        embeddings = []
        for vector in vectors:
            if isinstance(vector, torch.Tensor):
                stones = vector
            else:
                stones = torch.Tensor(vector)
            stones = stones.long().to(self.embeddings.device)
            black_mask = (stones == 0)
            white_mask = (stones == 1)
            if black_mask.long().sum() == white_mask.long().sum():
                stones[black_mask] = 1
                stones[white_mask] = 0
            stones = stones.view(-1, 1, 1).repeat(1, 1, self.dim)
            embeddings.append(torch.gather(
                self.embeddings.repeat(stones.size(0), 1, 1), 1, stones
            ))
        return F.normalize(torch.stack(embeddings, 0).squeeze(2), p=2, dim=-1)


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_dim, dim, radius=6, laplacian_matrix=None):
        super(GraphConvolutionLayer, self).__init__()
        if laplacian_matrix is None:
            laplacian_matrix = get_laplacian_matrix(radius)
        self.register_buffer('laplacian_matrix', laplacian_matrix)
        self.weight = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(in_dim, dim))
        )

    def forward(self, x):
        return torch.matmul(self.laplacian_matrix, torch.matmul(x, self.weight))


class GraphBatchNorm(nn.Module):

    def __init__(self, dim):
        super(GraphBatchNorm, self).__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)
    
    
class GraphInstanceNorm(nn.Module):
    
    def __init__(self, dim):
        super(GraphInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm1d(dim)
        
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)
    
    
class GraphResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_dim, dim, radius=6, 
                 gcn_cls=GraphConvolutionLayer):
        super(GraphResidualBlock, self).__init__()
        in_dim *= self.expansion
        out_dim = dim * self.expansion
        layers = []
        layers.append(nn.Linear(in_dim, dim, bias=False))
        # layers.append(GraphBatchNorm(dim))
        layers.append(GraphInstanceNorm(dim))
        layers.append(nn.ReLU())
        layers.append(gcn_cls(dim, dim, radius))
        # layers.append(GraphBatchNorm(dim))
        layers.append(GraphInstanceNorm(dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, out_dim, bias=False))
        # layers.append(GraphBatchNorm(out_dim))
        layers.append(GraphInstanceNorm(out_dim))
        self.stem = nn.Sequential(*layers)
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim), 
                # GraphBatchNorm(out_dim)
                GraphInstanceNorm(out_dim)
            )
        else:
            self.shortcut = None
        
    def forward(self, x):
        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return F.relu(self.stem(x) + shortcut)


class GraphConvolutionNetwork(NetworkBase):

    def __init__(
            self, in_dim, hidden_dim, block_num, 
            radius=6, gcn_cls=GraphConvolutionLayer
        ):
        super(GraphConvolutionNetwork, self).__init__()
        layers = []
        in_dim //= 4
        for _ in range(block_num):
            layers.append(GraphResidualBlock(in_dim, hidden_dim, radius, gcn_cls))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        in_dim *= GraphResidualBlock.expansion
        self.classifier = nn.Sequential(nn.Linear(in_dim, 1, bias=False))
    
    def forward(self, x):
        return self.classifier(self.backbone(x)).squeeze(-1)
               