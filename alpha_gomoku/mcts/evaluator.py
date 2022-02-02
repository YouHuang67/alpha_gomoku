import torch
import torch.nn.functional as F

from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board


class Evaluator(object):

    def __init__(self, net, size=Board.BOARD_SIZE):
        self.net = net
        self.size = size

    def __call__(self, board, actions=None):
        if board.is_over:
            if board.winner == board.player:
                return dict(), 1.0
            else:
                return dict(), 0.0

        board_tensor = torch.LongTensor(board.vector).reshape(1, self.size, -1)
        board_tensors = torch.cat([
            func(board_tensor) for func in utils.AUGMENTATION_FUNCS.values()
        ], dim=0)
        players = torch.Tensor([board.player]).expand(len(utils.AUGMENTATION_FUNCS))

        outs, values = self.net((board_tensors, players))
        outs = outs.cpu().detach().reshape(*board_tensors.size())
        out = torch.cat([
            func(out.unsqueeze(0)) for func, out in zip(utils.INVERSE_FUNCS.values(), outs)
        ], dim=0).mean(0).view(-1)

        if actions is None:
            legal_mask = (board_tensor.view(-1) == 2)
        else:
            actions = torch.LongTensor(list(map(self.flatten, actions)))
            legal_mask = torch.zeros(board_tensor.numel()).bool()
            legal_mask.scatter_(0, actions, torch.ones_like(actions).bool())

        legal_actions = torch.arange(board_tensor.numel())[legal_mask]
        probs = F.softmax(out[legal_mask], dim=0).cpu().numpy()

        value = torch.tanh(values.mean()).cpu().detach().numpy()

        return {self.unflatten(action.item()): prob
                for action, prob in zip(legal_actions, probs)}, value

    def flatten(self, action):
        return action[0] * self.size + action[1]

    def unflatten(self, action):
        return action // self.size, action % self.size





