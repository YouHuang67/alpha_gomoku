import torch
import torch.nn as nn
import torch.nn.functional as F
from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board


class PlayerEmbedding(nn.Module):

    def __init__(self, dim):
        super(PlayerEmbedding, self).__init__()
        self.dim = dim
        self.register_buffer('embeddings', F.normalize(
            torch.randn(Board.STONE_NUM, 3, dim), dim=-1, p=2
        ))

    def forward(self, boards):
        embeddings = []
        for board in utils.tolist(boards):
            stones = torch.Tensor(board.vector).long().to(self.embeddings.device)
            stones = stones.view(-1, 1, 1).repeat(1, 1, self.dim)
            embeddings.append(torch.gather(self.embeddings, 1, stones))
        return torch.stack(embeddings, 0).squeeze(2)


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


class GraphConvolutionNetwork(nn.Module):

    def __init__(
            self, hidden_dim, layer_num, 
            embedding=None, dim=None, radius=6
        ):
        super(GraphConvolutionNetwork, self).__init__()
        if embedding is None:
            assert dim is not None
            embedding = PlayerEmbedding(dim)
        self.embedding = embedding
        layers = []
        in_dim = dim
        for _ in range(layer_num - 1):
            layers.append(GraphConvolutionLayer(in_dim, hidden_dim, radius))
            layers.append(GraphBatchNorm(hidden_dim))
            in_dim = hidden_dim
        layers.append(GraphConvolutionLayer(in_dim, 1, radius))
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, boards):
        return self.backbone(self.embedding(boards)).squeeze(-1)