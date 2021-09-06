from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ... import utils
from ...train import SupervisedTrainer
from ...train import SupervisedPipelineBase
from ..models import Model
from ..models import network_classes
from ..models import embedding_classes


class VanillaGCNTrainer(SupervisedTrainer):

    @staticmethod
    def entropy(logits, mask):
        exp_logits = logits.exp()
        probs = exp_logits / exp_logits.sum(-1, keepdim=True)
        probs.masked_fill_(mask == 0, 1)
        log_probs = probs.log()
        return -log_probs.sum(-1) / mask.sum(-1)

    def loss(self, samples, training):
        model = self.models
        attack, defense, action = samples
        attack_tensor, attack_mask = attack
        attack_logits = model(attack_tensor)
        attack_logits.masked_fill_(attack_mask == 0, -float('inf'))
        action_loss = F.cross_entropy(attack_logits, action)
        defense_tensor, defense_mask = defense
        defense_logits = model(defense_tensor)
        defense_logits.masked_fill_(defense_mask == 0, -float('inf'))
        attack_entropy = self.entropy(attack_logits, attack_mask)
        defense_entropy = self.entropy(defense_logits, defense_mask)
        value_loss = (-attack_entropy + defense_entropy).mean()
        loss = action_loss + value_loss
        acc = (attack_logits.argmax(-1) == action).float().mean()
        return OrderedDict(loss=loss, act_loss=action_loss, 
                           val_loss=value_loss, acc=acc)
        
        
class GCNPipeline(SupervisedPipelineBase):
    
    def to_tensor(self, x):
        return self.models.to_tensor(x)
    
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