from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..datasets import VCTDataset
from .base import TrainerBase
from .base import PipelineBase


class SupervisedTrainer(TrainerBase):
    
    def loss(self, X1, X2, y):
        model = self.models
        prob, v1 = model(X1)
        _, v2 = model(X2)
        ce_loss = F.cross_entropy(prob, y)
        binary_loss = (F.binary_cross_entropy_with_logits(
            v1, torch.ones(y.size(0))
        ) + F.binary_cross_entropy_with_logits(
            v2, torch.zeros(y.size(0))
        )) / 2
        acc = (prob.argmax(-1) == y).float().mean()
        return ce_loss, binary_loss, acc
            
    def step(self, info=''):
        model = self.models.train()
        optimizer = self.optimizers
        ce_loss_meter = utils.AverageMeter()
        binary_loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        with tqdm(self.dataloaders[0], desc=info) as pbar:
            for samples in pbar:
                ce_loss, binary_loss, acc = self.loss(*samples)
                loss = ce_loss + binary_loss
                model.zero_grad()
                loss.backward()
                optimizer.step()
                ce_loss_meter.update(ce_loss.item(), samples[0].size(0))
                binary_loss_meter.update(binary_loss.item(), samples[0].size(0))
                acc_meter.update(acc.item(), samples[0].size(0))
                pbar.set_description(info + 
                                     f'ce_loss: {ce_loss_meter():.4f} ' + 
                                     f'binary_loss: {binary_loss_meter():.4f} ' + 
                                     f'acc: {acc_meter():.4f}')
        return OrderedDict(ce_loss=ce_loss_meter(), 
                           binary_loss=binary_loss_meter(), 
                           acc=acc_meter())
    
    @torch.no_grad()
    def eval(self):
        self.models.eval()
        ce_loss_meter = utils.AverageMeter()
        binary_loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        info = 'evaluate: '
        with tqdm(self.dataloaders[0], desc=info) as pbar:
            for samples in pbar:
                ce_loss, binary_loss, acc = self.loss(*samples)
                ce_loss_meter.update(ce_loss.item(), samples[0].size(0))
                binary_loss_meter.update(binary_loss.item(), samples[0].size(0))
                acc_meter.update(acc.item(), samples[0].size(0))
                pbar.set_description(info + 
                                     f'ce_loss: {ce_loss_meter():.4f} ' + 
                                     f'binary_loss: {binary_loss_meter():.4f} ' + 
                                     f'acc: {acc_meter():.4f}')
        return OrderedDict(eval_ce_loss=ce_loss_meter(), 
                           eval_binary_loss=binary_loss_meter(), 
                           eval_acc=acc_meter())
    
    
class SupervisedPipelineBase(PipelineBase):
    
    def to_tensor(self, actions):
        raise NotImplementedError
    
    def make_datasets(self):
        args = self.args
        root = Path(args.root)
        vct_path = root / 'vct_actions.json'
        if vct_path.is_file():
            dataset = VCTDataset(self.to_tensor)
            dataset.load(vct_path)
        else:
            dataset = VCTDataset(self.to_tensor, args.root)
            dataset.save(vct_path)
        return dataset.split(args.split)
    
    def make_optimizers(self):
        args = self.args
        optimizer_cls = utils.optim_cls_dict[args.optimizer.lower()]
        optimizer_args = utils.get_func_kwargs(optimizer_cls, args.__dict__)
        return optimizer_cls(self.models.parameters(), **optimizer_args)
    
    def train(self):
        args = self.args
        trainer = self.trainer
        losses = OrderedDict()
        for epoch in range(1, args.epochs+1):
            info = f'epoch {epoch:d}/{args.epochs}: '
            for lossname, lossval in trainer.step(info).items():
                losses.setdefault(lossname, list()).append(lossval)
            for lossname, lossval in trainer.eval().items():
                losses.setdefault(lossname, list()).append(lossval)
            self.save(losses)
        return losses
            
        