import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ..cppboard import Board
from .vct_dataset import PiskvorkVCTActions


class VCTDefenseActions(PiskvorkVCTActions):

    def __init__(self, dir=''):
        self.augmentation = False
        if dir:
            vct_path = Path(dir) / 'vct_actions.json'
            vct_defense_path = Path(dir) / 'vct_defense_actions.json'
            assert vct_path.is_file()
            self.load(vct_path)
            if vct_defense_path.is_file():
                self.load_defense(vct_defense_path)
            else:
                self.make_defense()
                self.save_defense(vct_defense_path)
        else:
            self.actions = []
            self.vct_actions = []
            self.vct_defense_actions = []

    def __len__(self):
        return len(self.vct_actions) + len(self.vct_defense_actions)

    def __getitem__(self, item):
        if item < len(self.vct_actions):
            return super(VCTDefenseActions, self).__getitem__(item)
        else:
            index, defense_actions = \
                self.vct_defense_actions[item - len(self.vct_actions)]
            actions, action = super(VCTDefenseActions, self).__getitem__(index)
            return [actions[0] + action], defense_actions

    def make_defense(self):
        self.vct_defense_actions = []
        for index in tqdm(range(len(self.vct_actions)), 'get vct defense actions'):
            actions, action = self[index]
            board = Board(actions[0] + action)
            if not board.is_over:
                defense_actions = board.evaluate(1)
                assert board.attacker != board.player
                if len(defense_actions):
                    self.vct_defense_actions.append((index, defense_actions))

    def save_defense(self, path):
        with open(path, 'w') as json_file:
            json.dump({'vct_defense_actions': self.vct_defense_actions}, json_file)

    def load_defense(self, path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        self.vct_defense_actions = [(index, list(map(tuple, acts)))
                                    for index, acts in data['vct_defense_actions']]

    def split(self, ratio, shuffle=True):
        sample_num = len(self.vct_actions)
        split = int(sample_num * ratio)
        assert 0 < split < sample_num
        if shuffle:
            indice = np.argsort(np.random.rand(sample_num)).tolist()
        else:
            indice = list(range(sample_num))
        actions = self.actions
        vct_actions = self.vct_actions

        first_set = self.__class__()
        first_set.actions = list(actions)
        first_set.vct_actions = [vct_actions[idx] for idx in indice[:split]]
        second_set = self.__class__()
        second_set.actions = list(actions)
        second_set.vct_actions = [vct_actions[idx] for idx in indice[split:]]

        mapping = [None] * sample_num
        for new_idx, idx in enumerate(indice):
            latter = new_idx >= split
            mapping[idx] = (new_idx - split * int(latter), int(latter))
        sets = [first_set, second_set]
        for index, defense_actions in self.vct_defense_actions:
            new_index, set_id = mapping[index]
            sets[set_id].vct_defense_actions.append((new_index, defense_actions))
        return first_set, second_set


