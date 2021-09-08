import torch
import torch.nn.functional as F

from .base import Model
from .base import get_adjacent_matrix
from .base import normalize_adjacent_matrix
from .residual import PlayerEmbedding
from .residual import GraphConvolutionLayer


class AuxiliaryEmbedding(PlayerEmbedding):
    
    def __init__(self, dim):
        super(AuxiliaryEmbedding, self).__init__(dim)
        self.register_buffer('auxiliary', torch.randn(dim))
        
    def forward(self, vectors):
        embedding = super(AuxiliaryEmbedding, self).forward(vectors)
        auxiliary = F.noramlize(self.auxiliary, p=2)
        auxiliary = auxiliary.view(1, 1, -1).repeat(embedding.size(0), 1, 1)
        return torch.cat([auxiliary, embedding], dim=1)
    
    
class AuxiliaryGraphConvolution(GraphConvolutionLayer):
    
    def __init__(self, in_dim, dim, radius=6):
        adjacent_matrix = get_adjacent_matrix(radius)
        size = adjacent_matrix.size(0)
        adjacent_matrix = torch.cat([
            torch.ones(size, 1), adjacent_matrix
        ], dim=1)
        adjacent_matrix = torch.cat([
            torch.ones(1, size + 1), adjacent_matrix
        ], dim=0)
        super(AuxiliaryGraphConvolution, self).__init__(
            in_dim, dim, 
            laplacian_matrix=normalize_adjacent_matrix(adjacent_matrix)
        )
        

class AuxiliaryModel(Model):
    
    def preprocess(self, vector):
        embedding, mask = super(AuxiliaryModel, self).preprocess(vector)
        mask = torch.cat([
            torch.ones(mask.size(0), 1).to(mask.device), mask
        ], dim=1)
        return embedding, mask