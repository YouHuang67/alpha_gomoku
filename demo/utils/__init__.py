from pathlib import Path

from .dihedral_reflection import *


ROOT_DIR = Path(__file__).resolve().parents[0]
WEIGHT_DIR = ROOT_DIR / 'weights'
BOARD_SIZE, STONE_NUM = 15, 15 ** 2
BLACK, WHITE, EMPTY = 0, 1, 2
