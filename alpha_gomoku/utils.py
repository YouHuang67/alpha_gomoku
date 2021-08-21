import os
import random
import numpy as np
from pathlib import Path
from collections import Iterable

import torch


ROOT = str(Path(__file__).parents[0])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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