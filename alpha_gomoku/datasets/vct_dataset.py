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
    
    def __init__(self, to_tensor, root='', augmentation=True, dir=None):
        super(VCTDataset, self).__init__(root, augmentation)
        self.to_tensor = to_tensor
        time.sleep(1)
        if dir is not None:
            dir = Path(dir)
            assert dir.is_dir()
        else:
            dir = Path(root) / '_temp_tensors' / utils.time_format()
            dir.mkdir(parents=True, exist_ok=False)
        self.dir = dir

    def prepare_samples(self, desc=''):
        for sample in tqdm(DataLoader(self, batch_size=1, shuffle=False), desc=desc):
            pass
        
    def __getitem__(self, item):
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
            torch.save(vectors, path)
        attack_vector, defense_vector, action = random.choice(vectors)
        attack = self.to_tensor(attack_vector)
        defense = self.to_tensor(defense_vector)
        return attack, defense, action
    
    def split(self, ratio, shuffle=True):
        return super(VCTDataset, self).split(
            ratio, shuffle, to_tensor=self.to_tensor, 
            root=self.root, augmentation=self.augmentation
        )