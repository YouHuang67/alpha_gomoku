from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import Model
from ..models import network_classes
from ..models import embedding_classes
from ... import utils
from ...train import SupervisedTrainer
from ...train import SupervisedPipelineBase


class VanillaGCNTrainer(SupervisedTrainer):
    
    def __init__(self, dataloaders, models, optimizers):
        super(VanillaGCNTrainer, self).__init__(
            dataloaders, models, optimizers, loss_func=nn.CrossEntropyLoss()
        )
        
        
class GCNPipeline(SupervisedPipelineBase):
    
    def to_tensor(self, actions):
        return self.models.to_tensor(actions)
    
    def make_dir(self):
        return str(Path(utils.ROOT) / 'data' / 'gcn' / utils.time_format())
    
    def make_models(self):
        args = self.args
        embedding_cls = embedding_classes[args.embedding.lower()]
        embedding = embedding_cls(
            **utils.get_func_kwargs(embedding_cls, args.__dict__)
        )
        network_cls = network_classes[args.network.lower()]
        network = network_cls(
            in_dim=embedding.dim, 
            **utils.get_func_kwargs(network_cls, args.__dict__)
        )
        return Model(embedding, network)
    
    def make_trainer(self):
        args = self.args
        batch_size = args.batch_size
        train_set, test_set = self.datasets
        dataloaders = (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False)
        )
        return VanillaGCNTrainer(dataloaders, self.models, self.optimizers)