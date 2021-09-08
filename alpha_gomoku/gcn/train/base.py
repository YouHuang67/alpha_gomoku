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
        attack_tensor = attack_tensor.to(self.device)
        attack_mask = attack_mask.to(self.device)
        action = action.to(self.device)
        attack_logits = model(attack_tensor)
        attack_logits.masked_fill_(attack_mask == 0, -float('inf'))
        action_loss = F.cross_entropy(attack_logits, action)
        defense_tensor, defense_mask = defense
        defense_tensor = defense_tensor.to(self.device)
        defense_mask = defense_mask.to(self.device)
        defense_logits = model(defense_tensor)
        defense_logits.masked_fill_(defense_mask == 0, -float('inf'))
        attack_entropy = self.entropy(attack_logits, attack_mask).detach()
        defense_entropy = self.entropy(defense_logits, defense_mask)
        value_loss = (-attack_entropy + defense_entropy).mean()
        loss = action_loss + self.value_weight * value_loss
        acc = (attack_logits.argmax(-1) == action).float().mean()
        entropy = torch.cat([attack_entropy, defense_entropy.detach()], dim=0)
        labels = torch.cat([torch.ones_like(attack_entropy).cpu(), 
                            torch.zeros_like(defense_entropy).cpu()], dim=0)
        auc = torch.Tensor([roc_auc_score(labels, entropy.clone().detach().cpu())])
        return OrderedDict(loss=loss, act_loss=action_loss, 
                           val_loss=value_loss, acc=acc, auc=auc)
        
        
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
        return Model(embedding, network).to(self.device)
    
    def make_trainer(self):
        args = self.args
        batch_size = args.batch_size
        train_set, test_set = self.datasets
        for sample in tqdm(
            DataLoader(train_set, batch_size=1, shuffle=False), 
            desc='preparing train samples...'
        ):
            pass
        for sample in tqdm(
            DataLoader(test_set, batch_size=1, shuffle=False), 
            desc='preparing test samples...'
        ):
            pass
        dataloaders = (
            DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16),
            DataLoader(test_set, batch_size=2*batch_size, shuffle=False, num_workers=16)
        )
        return VanillaGCNTrainer(
            dataloaders, self.models, self.optimizers, 
            value_weight=args.value_weight
        )