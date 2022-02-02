import json
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
            actions, action = super(VCTDefenseActions, self).__getitem__(item)
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


