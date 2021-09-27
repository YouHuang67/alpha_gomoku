from pathlib import Path
from argparse import Namespace
from collections import Iterable

import torch
import torch.nn as nn

from .. import utils


class TrainerBase(object):
    
    def __init__(self, dataloaders, models, optimizers, loss_func=None):
        self.dataloaders = dataloaders
        self.models = models
        self.optimizers = optimizers
        self._loss = loss_func
        self.device = utils.DEVICE
        
    def loss(self, *args, **kwargs):
        if self._loss is None:
            raise NotImplementedError
        return self._loss(*args, **kwargs)
    
    def step(self, desc=''):
        raise NotImplementedError
    
    @torch.no_grad()
    def eval(self):
        raise NotImplementedError


class PipelineBase(object):
    
    def __init__(self, dir=None, **kwargs):
        self.args = Namespace(**kwargs)
        self.device = utils.DEVICE
        if dir is None:
            self.dirs = self.make_dirs()
        else:
            self.args.dir = dir
            self.dirs = self.make_dirs()
            config = utils.json_load(self.dirs['config'])
            config.pop('root', None)
            self.args.__dict__.update(config)
        utils.set_seed(self.args.seed)
        self.datasets = self.make_datasets()
        self.models = self.make_models()
        if dir is not None:
            self.load_state_dict(torch.load(self.dirs['weights'], map_location='cpu'))
        self.optimizers = self.make_optimizers()
        self.trainer = self.make_trainer()
        
    def make_dir(self):
        raise NotImplementedError
        
    def make_dirs(self):
        args = self.args
        if not hasattr(args, 'dir'):
            args.dir = self.make_dir()
        dir = Path(args.dir)
        return {
            'dir': dir, 
            'config': dir / 'config.json', 
            'losses': dir / 'losses.json', 
            'weights': dir / 'weights.pth'
        }
        
    def make_datasets(self):
        raise NotImplementedError
    
    def make_models(self):
        raise NotImplementedError
    
    def make_optimizers(self):
        raise NotImplementedError
    
    def make_trainer(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def state_dict(self):
        def get_state_dict(m):
            if isinstance(m, nn.Module):
                return m.state_dict()
            elif isinstance(m, dict):
                return {k: get_state_dict(v) for k, v in m.items()}
            elif isinstance(m, Iterable):
                return [get_state_dict(i) for i in m]
            else:
                raise Exception
        return get_state_dict(self.models)
    
    def load_state_dict(self, state_dict):
        def load_state_dict(m, s):
            if isinstance(m, nn.Module):
                m.load_state_dict(s)
            elif isinstance(m, dict):
                for k, v in m.items():
                    load_state_dict(v, s[k])
            elif isinstance(m, Iterable):
                for i, j in zip(m, s):
                    load_state_dict(i, j)
            else:
                raise Exception
        load_state_dict(self.models, state_dict)
        
    def config(self):
        return self.args.__dict__
        
    def save(self, losses=None):
        dirs = self.dirs
        Path(dirs['dir']).mkdir(parents=True, exist_ok=True)
        utils.json_save(dirs['config'], self.config())
        if losses is not None:
            utils.json_save(dirs['losses'], losses)
        torch.save(self.state_dict(), dirs['weights'])
        