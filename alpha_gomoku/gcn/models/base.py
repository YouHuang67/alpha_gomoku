import torch
import torch.nn as nn
from torch.nn.functional import embedding

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
        
    def preprocess(self, vector):
        embedding = self.embedding(vector)
        mask = torch.zeros_like(vector).float()
        mask[vector == Board.EMPTY] = 1
        return embedding, mask
    
    def forward(self, x):
        embedding, mask = self.preprocess(x)
        return self.network(embedding), mask
    
    
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


def normalize_adjacent_matrix(adjacent_matrix):
    degrees_rsqrt = adjacent_matrix.sum(-1).rsqrt()
    return torch.outer(degrees_rsqrt, degrees_rsqrt) * adjacent_matrix


def get_laplacian_matrix(radius):
    return normalize_adjacent_matrix(get_adjacent_matrix(radius))