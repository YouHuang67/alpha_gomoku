import torch
import torch.nn as nn

from ...cppboard import Board


class EmbeddingBase(nn.Module):
    pass


class NetworkBase(nn.Module):
    pass


class Model(nn.Module):
    
    def __init__(self, embdding, network):
        super(Model, self).__init__()
        self.embedding = embdding
        self.network = network
        
    def to_tensor(self, actions):
        return self.embedding(Board(actions))[0]
    
    def forward(self, x):
        return self.network(x)