import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parents[3]))
from alpha_gomoku.cppboard import Board
from alpha_gomoku.gcn import models


def main():
    gcn = models.GraphConvolutionNetwork(128, 4, dim=128)
    print(gcn([Board(), Board()]).shape)


if __name__ == '__main__':
    main()