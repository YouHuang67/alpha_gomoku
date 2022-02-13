from pathlib import Path

from .dihedral_reflection import *


ROOT_DIR = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT_DIR / 'weights'
SIZE, NUM_STONES = 15, 15 ** 2
BLACK, WHITE, EMPTY = 0, 1, 2


def flatten(*args):
    row, col = (args[0] if len(args) == 1 else args)
    return row * SIZE + col


def unflatten(action):
    return action // SIZE, action % SIZE
