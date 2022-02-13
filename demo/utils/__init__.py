from pathlib import Path

from .dihedral_reflection import *


ROOT_DIR = Path(__file__).resolve().parents[0]
WEIGHT_DIR = ROOT_DIR / 'weights'
BOARD_SIZE = 15
STONE_NUM = BOARD_SIZE ** 2
BLACK = 0
WHITE = 1
EMPTY = 2
OPEN_FOUR = 1
FOUR = 2
OPEN_THREE = 3
THREE = 4
OPEN_TWO = 5