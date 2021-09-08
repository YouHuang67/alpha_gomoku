from numpy import isin
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
    
    def loss(self, samples, training):
        raise NotImplementedError

    @staticmethod
    def get_sample_size(samples):
        if isinstance(samples, torch.Tensor):
            x = samples
        elif isinstance(samples, dict):
            x = next(iter(samples.values()))
        elif isinstance(samples, (list, tuple)):
            x = samples[0]
        else:
            raise Exception(f'cannot get sample size of instance of {type(samples)}')
        if isinstance(x, torch.Tensor):
            return x.size(0)
        elif isinstance(x, (list, tuple)):
            return len(x)
        else:
            raise Exception(f'cannot get size of instance of {type(x)}')
            
    def step(self, desc=''):
        model = self.models.train()
        optimizer = self.optimizers
        meters = OrderedDict()
        with tqdm(self.dataloaders[0], desc=desc) as pbar:
            for samples in pbar:
                losses = self.loss(samples, training=True)
                model.zero_grad()
                losses['loss'].backward()
                optimizer.step()
                sample_size = self.get_sample_size(samples)
                info = ''
                for lossname, lossval in losses.items():
                    meter = meters.setdefault(lossname, utils.AverageMeter())
                    meter.update(lossval.item(), sample_size)
                    info += f' {lossname}: {meter():.4f}'
                pbar.set_description(desc + info)
        return OrderedDict(**{lossname: meter() for lossname, meter in meters.items()})
    
    @torch.no_grad()
    def eval(self):
        self.models.eval()
        meters = OrderedDict()
        desc = 'evaluate: '
        with tqdm(self.dataloaders[1], desc=desc) as pbar:
            for samples in pbar:
                losses = self.loss(samples, training=False)
                sample_size = self.get_sample_size(samples)
                info = ''
                for lossname, lossval in losses.items():
                    meter = meters.setdefault(lossname, utils.AverageMeter())
                    meter.update(lossval.item(), sample_size)
                    info += f' {lossname}: {meter():.4f}'
                pbar.set_description(desc + info)
        return OrderedDict(**{lossname: meter() for lossname, meter in meters.items()})
    
    
class SupervisedPipelineBase(PipelineBase):
    
    def make_datasets(self):
        args = self.args
        root = Path(args.root)
        vct_path = root / 'vct_actions.json'
        if vct_path.is_file():
            dataset = VCTDataset()
            dataset.load(vct_path)
        else:
            dataset = VCTDataset(args.root)
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
            desc = f'epoch {epoch:d}/{args.epochs}: '
            for lossname, lossval in trainer.step(desc).items():
                losses.setdefault(lossname, list()).append(lossval)
            for lossname, lossval in trainer.eval().items():
                losses.setdefault(lossname, list()).append(lossval)
            self.save(losses)
        return losses
            
        