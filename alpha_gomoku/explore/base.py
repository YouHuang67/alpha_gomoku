import random

import torch

from .. import utils
from .. import datasets
from ..cppboard import Board


DATA_DIR = utils.DATA_DIR / 'explore'
LOG_DIR = utils.LOG_DIR / 'explore'


class VCTActions(torch.utils.data.Dataset):
    size = Board.BOARD_SIZE

    def __init__(self, augmentation=False, dataset=None, get_value=False):
        self.augmentation = augmentation
        self.get_value = get_value
        if dataset is None:
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

        if self.get_value:
            actions_num = len(board_actions)
            board_actions = board_actions[:random.randint(1, actions_num)]
            target = 1 - ((actions_num - len(board_actions)) % 2)
            weight = float(len(board_actions)) / float(actions_num)

        indices = torch.LongTensor(list(zip(*board_actions)))
        values = torch.LongTensor([(i % 2) - 2 for i in range(len(board_actions))])
        size = torch.Size([self.size, self.size])
        board = torch.sparse.LongTensor(indices, values, size).to_dense() + 2
        player = len(board_actions) % 2

        if self.get_value:
            return board, target, player, weight
        else:
            return board, self.action_flatten(*vct_action[index]), player

    @classmethod
    def action_flatten(cls, row, col):
        return row * cls.size + col

    def __len__(self):
        return len(self.dataset)

    def split(self, ratio):
        first_set, second_set = self.dataset.split(ratio)
        return self.__class__(augmentation=self.augmentation, dataset=first_set), \
               self.__class__(augmentation=self.augmentation, dataset=second_set)






