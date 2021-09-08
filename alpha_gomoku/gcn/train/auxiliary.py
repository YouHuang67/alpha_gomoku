from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ... import utils
from ..models import AuxiliaryModel
from ..models import AuxiliaryEmbedding
from ..models import GraphConvolutionNetwork
from ..models import AuxiliaryGraphConvolution
from .base import GCNPipeline
from .base import VanillaGCNTrainer
from .base import SupervisedPipelineBase


class AuxiliaryTrainer(VanillaGCNTrainer):
    
    def loss(self, samples, training):
        model = self.models
        attack, defense, action = samples
        attack = attack.to(self.device)
        defense = defense.to(self.device)
        action = action.to(self.device) + 1

        attack_logits, attack_masks = model(attack)
        attack_logits.masked_fill_(attack_masks == 0, -float('inf'))
        action_loss = F.cross_entropy(attack_logits, action)
        
        defense_logits, defense_masks = model(defense)
        defense_logits.masked_fill_(defense_masks == 0, -float('inf'))
        
        virtual_labels = torch.zeros_like(action)
        attack_value = F.cross_entropy(
            attack_logits, virtual_labels, reduction='none'
        ).detach()
        defense_value = F.cross_entropy(
            defense_logits, virtual_labels, reduction='none'
        )
        value_loss = (-attack_value + defense_value).mean()
        
        loss = action_loss + self.value_weight * value_loss
        acc = (attack_logits.argmax(-1) == action).float().mean()
        entropy = torch.cat([attack_value.clone().detach(), 
                             defense_value.clone().detach()], dim=0).cpu()
        labels = torch.cat([torch.ones_like(attack_value).cpu(), 
                            torch.zeros_like(defense_value).cpu()], dim=0)
        auc = torch.Tensor([roc_auc_score(labels, entropy)])
        return OrderedDict(loss=loss, act_loss=action_loss, 
                           val_loss=value_loss, acc=acc, auc=auc)
        
        
class AuxiliaryPipeline(SupervisedPipelineBase):
    
    def make_dir(self):
        return str(
            Path(utils.ROOT) / 'data' / 'auxiliary' / utils.time_format()
        )
    
    def make_models(self):
        args = self.args
        embedding = AuxiliaryEmbedding(
            **utils.get_func_kwargs(AuxiliaryEmbedding, args.__dict__)
        )
        network = GraphConvolutionNetwork(
            in_dim=embedding.dim, 
            gcn_cls=AuxiliaryGraphConvolution, 
            **utils.get_func_kwargs(GraphConvolutionNetwork, args.__dict__)
        )
        return AuxiliaryModel(embedding, network).to(self.device)
    
    def make_trainer(self):
        args = self.args
        batch_size = args.batch_size
        train_set, test_set = self.datasets
        if hasattr(args, 'train_dir'):
            train_set.dir = Path(args.train_dir)
        if not train_set.dir.is_dir():
            train_set.prepare_samples('prepare train samples: ')
        if hasattr(args, 'test_dir'):
            test_set.dir = Path(args.test_dir)
        if not test_set.dir.is_dir():
            test_set.prepare_samples('prepare test samples: ')
        dataloaders = (
            DataLoader(train_set, batch_size=batch_size, 
                       shuffle=True, num_workers=args.num_workers),
            DataLoader(test_set, batch_size=2*batch_size, 
                       shuffle=False, num_workers=args.num_workers)
        )
        return AuxiliaryTrainer(
            dataloaders, self.models, self.optimizers, 
            value_weight=args.value_weight
        )
        