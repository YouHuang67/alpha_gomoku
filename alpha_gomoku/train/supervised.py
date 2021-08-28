from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch

from .. import utils
from ..datasets import VCTDataset
from .base import TrainerBase
from .base import PipelineBase


class SupervisedTrainer(TrainerBase):
    
    def step(self, info=''):
        model = self.models.train()
        optimizer = self.optimizers
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        with tqdm(self.dataloaders[0], desc=info) as pbar:
            for X, y in pbar:
                pred = model(X)
                loss = self.loss(pred, y)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item(), X.size(0))
                acc_meter.update(
                    (pred.argmax(-1) == y).float().mean().item(), X.size(0)
                )
                pbar.set_description(info + 
                                     f'loss: {loss_meter():.4f} ' + 
                                     f'acc: {acc_meter():.4f}')
        return OrderedDict(loss=loss_meter(), acc=acc_meter())
    
    @torch.no_grad()
    def eval(self):
        model = self.models.eval()
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        info = 'evaluate: '
        with tqdm(self.dataloaders[0], desc=info) as pbar:
            for X, y in pbar:
                pred = model(X)
                loss = self.loss(pred, y)
                loss_meter.update(loss.item(), X.size(0))
                acc_meter.update(
                    (pred.argmax(-1) == y).float().mean().item(), X.size(0)
                )
                pbar.set_description(info + 
                                     f'eval loss: {loss_meter():.4f} ' + 
                                     f'eval acc: {acc_meter():.4f}')
        return OrderedDict(eval_loss=loss_meter(), eval_acc=acc_meter())
    
    
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
            
        