import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ..cppboard import Board
from .vct import get_vct_actions


def load_piskvork_record(path):
    actions = []
    with open(path, 'r') as file:
        for line in file.readlines()[2:-1]:
            row, col = [int(c) for c in line.strip().split(',')][:2]
            actions.append((row, col))
    return actions


def load_piskvork_records(root):
    for path in Path(root).rglob('*.rec'):
        yield load_piskvork_record(path)


class PiskvorkVCTActions(object):

    def __init__(self, root=None, augmentation=True):
        if root is None:
            self.actions = []
            self.vct_actions = []
        else:
            self.actions, self.vct_actions = \
                self.get_vct_actions_from_piskvork_records(root)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.vct_actions)

    def __getitem__(self, item):
        index, step, action = self.vct_actions[item]
        actions = self.actions[index]
        if self.augmentation:
            action_list = list(zip(*[
                Board.get_homogenous_actions(act) for act in actions
            ]))
            index = random.choice(range(len(action_list)))
            actions = action_list[index]
            action = Board.get_homogenous_actions(action)[index]
        return actions[:step], action

    @staticmethod
    def get_vct_actions_from_piskvork_records(root):
        pre_keys = []
        pre_actions = []
        for acts in load_piskvork_records(root):
            acts = list(zip(
                *[Board.get_homogenous_actions(act) for act in acts]
            ))
            pre_keys.extend([Board(ats).key for ats in acts])
            pre_actions.extend(acts)
        keys = set()
        actions = []
        for index, (key, acts) in enumerate(zip(pre_keys, pre_actions)):
            if index % 8 == 0:
                if key not in keys:
                    actions.append(acts)
            keys.add(key)
        vct_actions = []
        for index, acts in tqdm(
            list(enumerate(actions)), desc='get vct actions: '
        ):
            for step, act in get_vct_actions(acts):
                vct_actions.append((index, step, act))
        return actions, vct_actions

    def save(self, path):
        with open(path, 'w') as json_file:
            json.dump({'actions': self.actions, 
                       'vct_actions': self.vct_actions}, json_file)
    
    def load(self, path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        self.actions = [list(map(tuple, acts)) for acts in data['actions']]
        self.vct_actions = [(index, step, tuple(act))
                            for index, step, act in data['vct_actions']]
        
    def split(self, ratio, shuffle=True, **kwargs):
        sample_num = len(self)
        split = int(sample_num * ratio)
        assert 0 < split < sample_num
        if shuffle:
            indice = np.argsort(np.random.rand(sample_num)).tolist()
        else:
            indice = list(range(sample_num))
        actions = self.actions
        vct_actions = self.vct_actions
        first_set = self.__class__(**kwargs)
        first_set.actions = list(actions)
        first_set.vct_actions = [vct_actions[idx] for idx in indice[:split]]
        second_set = self.__class__(**kwargs)
        second_set.actions = list(actions)
        second_set.vct_actions = [vct_actions[idx] for idx in indice[split:]]
        return first_set, second_set