import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import utils
from ...cppboard import Board
from .base import NetworkBase
from .base import EmbeddingBase


class PlayerEmbedding(EmbeddingBase):

    def __init__(self, dim):
        super(PlayerEmbedding, self).__init__()
        self.dim = dim
        self.register_buffer('embeddings', 
            torch.randn(1, 3, dim)
        )

    def forward(self, boards):
        embeddings = []
        for board in utils.tolist(boards):
            stones = torch.Tensor(board.vector).long().to(self.embeddings.device)
            if board.player == Board.WHITE:
                black_mask = (stones == 0)
                white_mask = (stones == 1)
                stones[black_mask] = 1
                stones[white_mask] = 0
            stones = stones.view(-1, 1, 1).repeat(1, 1, self.dim)
            embeddings.append(torch.gather(
                self.embeddings.repeat(stones.size(0), 1, 1), 1, stones
            ))
        return F.normalize(torch.stack(embeddings, 0).squeeze(2), p=2, dim=-1)


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_dim, dim, radius=6):
        super(GraphConvolutionLayer, self).__init__()
        adjacent_matrix = self.get_adjacent_matrix(radius)
        degrees_rsqrt = adjacent_matrix.sum(-1).rsqrt()
        self.register_buffer('laplacian_matrix', 
            torch.outer(degrees_rsqrt, degrees_rsqrt) * adjacent_matrix
        )
        self.weight = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(in_dim, dim))
        )

    def forward(self, x):
        return torch.matmul(self.laplacian_matrix, torch.matmul(x, self.weight))

    @staticmethod
    def get_adjacent_matrix(radius):
        matrix = torch.zeros(Board.STONE_NUM, Board.STONE_NUM)
        for i in range(Board.STONE_NUM):
            x1, y1 = i // Board.BOARD_SIZE, i % Board.BOARD_SIZE
            for j in range(Board.STONE_NUM):
                x2, y2 = j // Board.BOARD_SIZE, j % Board.BOARD_SIZE
                if x1 == x2 and abs(y1 - y2) <= radius:
                    matrix[i, j] = 1
                elif y1 == y2 and abs(x1 - x2) <= radius:
                    matrix[i, j] = 1
                elif abs(x1 - x2) == abs(y1 - y2) <= radius:
                    matrix[i, j] = 1
        return matrix.float()


class GraphBatchNorm(nn.Module):

    def __init__(self, dim):
        super(GraphBatchNorm, self).__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)
    
    
class GraphResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_dim, dim, radius=6):
        super(GraphResidualBlock, self).__init__()
        in_dim *= self.expansion
        out_dim = dim * self.expansion
        layers = []
        layers.append(nn.Linear(in_dim, dim, bias=False))
        layers.append(GraphBatchNorm(dim))
        layers.append(nn.ReLU())
        layers.append(GraphConvolutionLayer(dim, dim, radius))
        layers.append(GraphBatchNorm(dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, out_dim, bias=False))
        layers.append(GraphBatchNorm(out_dim))
        self.stem = nn.Sequential(*layers)
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim), GraphBatchNorm(out_dim)
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
            self, in_dim, hidden_dim, block_num, radius=6
        ):
        super(GraphConvolutionNetwork, self).__init__()
        layers = []
        in_dim //= 4
        for _ in range(block_num):
            layers.append(GraphResidualBlock(in_dim, hidden_dim, radius))
            in_dim = hidden_dim
        layers.append(GraphConvolutionLayer(
            in_dim * GraphResidualBlock.expansion, 1, radius
        ))
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.backbone(x).squeeze(-1)