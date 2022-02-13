import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SIZE, NUM_STONES
from .utils import REFLECTION_FUNCS, INVERSE_FUNCS
from .utils import unflatten


class Vicinity(nn.Module):

    def __init__(self, radius=3):
        super(Vicinity, self).__init__()
        self.conv = nn.Conv2d(1, 1, (radius * 2 + 1), 1, radius, bias=False)
        nn.init.ones_(self.conv.weight)
        self.conv.weight.requires_grad_(False)

    @torch.no_grad()
    def forward(self, mask):
        if not mask.any():
            mask[SIZE // 2, SIZE // 2] = True
        mask = mask.float().view(1, 1, SIZE, SIZE)
        return (self.conv(mask) > 0.0).view(SIZE, SIZE).detach()


class NetWrapper(nn.Module):

    def __init__(self, model, gap=3):
        super(NetWrapper, self).__init__()
        self.model = model
        self.vicinity = Vicinity(gap)

    @staticmethod
    def actions_to_tensor(actions):
        indices = torch.LongTensor(actions).transpose(0, 1)
        values = torch.ones(len(actions)).long()
        size = torch.Size([SIZE, SIZE])
        return torch.sparse_coo_tensor(indices, values, size).to_dense()

    @staticmethod
    def board_to_tensor(board):
        return torch.LongTensor(board.vector).view(SIZE, SIZE), \
               torch.LongTensor(board.player)

    @torch.no_grad()
    def forward(self, board):
        board_tensor, player = self.board_to_tensor(board)
        legal_mask = (board_tensor == 2)
        index = random.randint(0, len(REFLECTION_FUNCS) - 1)
        func, inv_func = REFLECTION_FUNCS[index], INVERSE_FUNCS[index]
        logit, value = self.model((func(board_tensor.unsqueeze(0)), player))
        logit = inv_func(logit).view(SIZE, SIZE).detach()

        logit[~legal_mask] = -torch.inf
        probs = F.softmax(logit.reshape(NUM_STONES), 0)

        mask = legal_mask & self.vicinity(~legal_mask)
        actions = board.copy().evaluate(1)
        if len(actions):
            mask[~self.actions_to_tensor(actions).bool()] = False
        mask = mask.view(NUM_STONES)
        probs /= (mask.float() * probs).sum()

        return {unflatten(act): probs[act].item() for act in
                torch.where(mask)[0].numpy().tolist()}, value.item()
