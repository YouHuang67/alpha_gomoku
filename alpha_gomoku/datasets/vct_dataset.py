import time
import random
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import utils
from ..cppboard import Board
from .piskvork import PiskvorkVCTActions


class VCTDataset(PiskvorkVCTActions):
    
    def __init__(self, root='', augmentation=True, dir=None, load_all_samples=False):
        super(VCTDataset, self).__init__(root, augmentation)
        time.sleep(1)
        if dir is not None:
            dir = Path(dir)
            assert dir.is_dir()
        else:
            dir = Path(root) / '_temp_tensors' / utils.time_format()
        self.dir = dir
        self.load_all_samples = load_all_samples
        self.vectors = dict()

    def prepare_samples(self, desc=''):
        for sample in tqdm(DataLoader(self, batch_size=1, shuffle=False), desc=desc):
            pass
        
    def __getitem__(self, item):
        if self.load_all_samples and item in self.vectors:
            vectors = self.vectors[item]
        else:
            path = self.dir / f'{item}.pth'
            if path.is_file():
                vectors = torch.load(path, map_location='cpu')
            else:
                vectors = []
                for actions, vct_action in zip(*super(VCTDataset, self).__getitem__(item)):
                    board = Board(actions)
                    attack_vector = board.vector
                    board.move(vct_action)
                    defense_vector = board.vector
                    action = vct_action[0] * Board.BOARD_SIZE + vct_action[1]
                    vectors.append((attack_vector, defense_vector, action))
                self.dir.mkdir(parents=True, exist_ok=True)
                torch.save(vectors, path)
            if self.load_all_samples:
                self.vectors[item] = vectors
        attack_vector, defense_vector, action = random.choice(vectors)
        attack = torch.Tensor(attack_vector)
        defense = torch.Tensor(defense_vector)
        return attack, defense, action
    
    def split(self, ratio, shuffle=True):
        return super(VCTDataset, self).split(
            ratio, shuffle, root=self.root, 
            augmentation=self.augmentation, 
            load_all_samples=self.load_all_samples
        )