import torch
import torch.nn as nn

from ...cppboard import Board


def initialize(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return model


class EmbeddingBase(nn.Module):
    pass


class NetworkBase(nn.Module):
    pass


class Model(nn.Module):
    
    def __init__(self, embdding, network):
        super(Model, self).__init__()
        self.embedding = embdding
        self.network = network
        
    def to_tensor(self, x):
        if isinstance(x, Board):
            vector = x.vector
        elif isinstance(x[0], int):
            vector = x
        else:
            vector = Board(x).vector
        vector = torch.Tensor(vector)
        embedding = self.embedding([vector])[0]
        mask = torch.zeros_like(vector).float()
        mask[vector == Board.EMPTY] = 1
        return embedding, mask
    
    def forward(self, x):
        return self.network(x)
    
    
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


def get_laplacian_matrix(radius):
    adjacent_matrix = get_adjacent_matrix(radius)
    degrees_rsqrt = adjacent_matrix.sum(-1).rsqrt()
    return torch.outer(degrees_rsqrt, degrees_rsqrt) * adjacent_matrix