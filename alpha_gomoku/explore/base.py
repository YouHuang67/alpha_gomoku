import random

import torch

from .. import utils
from .. import datasets
from ..cppboard import Board


DATA_DIR = utils.DATA_DIR / 'explore'
LOG_DIR = utils.LOG_DIR / 'explore'


class VCTActions(torch.utils.data.Dataset):
    size = Board.BOARD_SIZE

    def __init__(self, augmentation=False):
        self.augmentation = augmentation
        dataset_path = utils.DATA_DIR / 'gomocup' / 'records' / 'vct_actions.json'
        assert dataset_path.is_file()
        dataset = datasets.PiskvorkVCTActions(augmentation=True)
        dataset.load(dataset_path)
        self.dataset = dataset

    def __getitem__(self, item):
        board_actions, vct_action = self.dataset[item]
        if self.augmentation:
            index = random.randint(0, len(board_actions) - 1)
        else:
            index = 0
        board_actions = board_actions[index]
        vct_action = vct_action[index]
        indices = torch.LongTensor(list(zip(*board_actions)))
        values = torch.LongTensor([(i % 2) - 2 for i in range(len(board_actions))])
        size = torch.Size([self.size, self.size])
        board = torch.sparse.LongTensor(indices, values, size).to_dense() + 2
        return board, Board.action_flatten(*vct_action)

    def __len__(self):
        return len(self.dataset)






