import os
import json
import time
import random
import inspect
import numpy as np
from pathlib import Path
from collections import Iterable

import torch


ROOT = str(Path(__file__).parents[0])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def str2bool(x):
    if 't' in x.lower():
        return True
    elif 'f' in x.lower():
        return False
    else:
        raise Exception(f'Illegal {x} for boolean parameter')


tolist = lambda x: x if isinstance(x, Iterable) else [x]


def json_load(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)
    
    
def json_save(path, obj):
    with open(path, 'w') as json_file:
        json.dump(obj, json_file)


def args_to_dirname(args, keyargs):
    keyargs = sorted(list(keyargs))

    def format(x):
        if isinstance(x, bool):
            return 'true' if x else 'false'
        elif isinstance(x, int):
            return f'{x:d}'
        elif isinstance(x, float):
            if x > 1e-2:
                return f'{x:.2f}'
            else:
                return f'{x:.2e}'
        elif isinstance(x, str):
            return f'{x}'
        else:
            raise ValueError(f'Illegal type {type(x)} of {x}')
    return '-'.join(f'{k}_{format(args.__dict__[k])}' for k in keyargs)


def time_format():
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())


def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


optim_cls_dict = {
    op.__name__.lower(): op for _, op in inspect.getmembers(torch.optim)
    if isinstance(op, type) and issubclass(op, torch.optim.Optimizer)
}


def set_hub(hubdir):
    if hubdir:
        torch.hub.set_dir(hubdir)


def get_func_kwargs(func, kwargs):
    return {
        arg: kwargs[arg]
        for arg in inspect.getfullargspec(func).args
        if arg in kwargs
    }
    
    
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def __call__(self):
        assert self.count
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, vals, n=1):
        if isinstance(vals, Iterable):
            for val in vals:
                self.sum += val
                self.count += 1
        else:
            self.sum += vals * n
            self.count += n