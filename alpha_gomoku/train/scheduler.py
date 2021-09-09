from torch import optim
from torch.optim.lr_scheduler import MultiStepLR


def get_scheduler(optimizer):
    if isinstance(optimizer, optim.SGD):
        return MultiStepLR(optimizer, milestones=[60, 90, 120, 150], gamma=0.1)
    else:
        return None