import os
import sys
import time
import random
import argparse
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.realpath('.'))
from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.datasets import piskvork
from alpha_gomoku.datasets import vct
from alpha_gomoku.testing import show_vct


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--dir', type=str, default='D:/hy/projects/alpha_gomoku/alpha_gomoku/data/gomocup/records')
    return parser.parse_args()


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    vct_action_path = Path(args.dir) / 'vct_actions.json'
    if vct_action_path.is_file():
        vct_action_dataset = piskvork.PiskvorkVCTActions(augmentation=False)
        vct_action_dataset.load(Path(args.dir) / 'vct_actions.json')
    else:
        vct_action_dataset = piskvork.PiskvorkVCTActions(Path(args.dir), False)
    for index in tqdm(range(len(vct_action_dataset)), 'check legality of vct actions: '):
        actions, vct_action = vct_action_dataset[index]
        board = Board(actions)
        if board.is_legal(vct_action):
            attack_actions = board.evaluate()
            if board.attacker not in [Board.BLACK, Board.WHITE]:
                print(f'\n{index}', board, vct_action, attack_actions)
        else:
            print(f'\n index {index} illegal action')
            print(board, vct_action)
            print('evaluated vct action: ', board.evaluate()[0])


if __name__ == '__main__':
    main()
