from .vct import *
from .piskvork import *
from ..cppboard import Board


class VCTDataset(PiskvorkVCTActions):
    
    def __init__(self, to_tensor, root=None, augmentation=True):
        super(VCTDataset, self).__init__(root, augmentation)
        self.to_tensor = to_tensor
        
    def __getitem__(self, item):
        actions, vct_action = super(VCTDataset, self).__getitem__(item)
        board = Board(actions)
        attack = self.to_tensor(board)
        board.move(vct_action)
        defense = self.to_tensor(board)
        return attack, defense, vct_action[0] * Board.BOARD_SIZE + vct_action[1]
    
    def split(self, ratio, shuffle=True):
        return super(VCTDataset, self).split(
            ratio, shuffle, to_tensor=self.to_tensor
        )