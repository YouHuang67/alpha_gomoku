import os
import random
import inspect
import numpy as np
from collections import Iterable

import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def str2bool(x):
    if 't' in x.lower():
        return True
    elif 'f' in x.lower():
        return False
    else:
        raise Exception(f'Illegal {x} for boolean parameter')


tolist = lambda x: x if isinstance(x, Iterable) else [x]


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

