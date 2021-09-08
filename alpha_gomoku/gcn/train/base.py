from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

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
    
    def __init__(self, dataloaders, models, optimizers, value_weight=1.0):
        super(VanillaGCNTrainer, self).__init__(
            dataloaders, models, optimizers
        )
        self.value_weight = value_weight

    @staticmethod
    def entropy(logits, masks):
        exp_logits = logits.exp()
        probs = exp_logits / exp_logits.sum(-1, keepdim=True)
        probs.masked_fill_(masks == 0, 1)
        log_probs = probs.log()
        return -log_probs.sum(-1) / masks.sum(-1)

    def loss(self, samples, training):
        model = self.models
        attack, defense, action = samples
        attack = attack.to(self.device)
        defense = defense.to(self.device)
        action = action.to(self.device)

        attack_logits, attack_masks = model(attack)
        attack_logits.masked_fill_(attack_masks == 0, -float('inf'))
        action_loss = F.cross_entropy(attack_logits, action)
        attack_entropy = self.entropy(attack_logits, attack_masks).detach()

        defense_logits, defense_masks = model(defense)
        defense_logits.masked_fill_(defense_masks == 0, -float('inf'))
        defense_entropy = self.entropy(defense_logits, defense_masks)
        value_loss = (-attack_entropy + defense_entropy).mean()

        loss = action_loss + self.value_weight * value_loss
        acc = (attack_logits.argmax(-1) == action).float().mean()
        entropy = torch.cat([attack_entropy.clone().detach(), 
                             defense_entropy.clone().detach()], dim=0).cpu()
        labels = torch.cat([torch.ones_like(attack_entropy).cpu(), 
                            torch.zeros_like(defense_entropy).cpu()], dim=0)
        auc = torch.Tensor([roc_auc_score(labels, entropy)])
        return OrderedDict(loss=loss, act_loss=action_loss, 
                           val_loss=value_loss, acc=acc, auc=auc)
        
        
class GCNPipeline(SupervisedPipelineBase):
    
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
        return Model(embedding, network).to(self.device)
    
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
        return VanillaGCNTrainer(
            dataloaders, self.models, self.optimizers, 
            value_weight=args.value_weight
        )