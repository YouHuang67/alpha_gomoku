import time
import random
from pathlib import Path

import torch

from .. import utils
from ..cppboard import Board
from .piskvork import PiskvorkVCTActions


class VCTDataset(PiskvorkVCTActions):
    
    def __init__(self, to_tensor, root='', augmentation=True):
        super(VCTDataset, self).__init__(root, augmentation)
        self.to_tensor = to_tensor
        time.sleep(1)
        dir = Path(root) / '_temp_tensors' / utils.time_format()
        dir.mkdir(parents=True, exist_ok=False)
        self.dir = dir
        
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