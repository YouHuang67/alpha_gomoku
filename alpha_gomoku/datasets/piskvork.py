import json
from tqdm import tqdm
from pathlib import Path

from alpha_gomoku.cppboard import Board
from alpha_gomoku.datasets.vct import get_vct_actions


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

    def __init__(self, root=None):
        if root is None:
            self.actions = []
            self.vct_actions = []
        else:
            self.actions, self.vct_actions = \
                self.get_vct_actions_from_piskvork_records(root)

    def __len__(self):
        return len(self.vct_actions)

    def __getitem__(self, item):
        index, step, action = self.vct_actions[item]
        actions = self.actions[index]
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