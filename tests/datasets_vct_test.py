import os
import sys

sys.path.append(os.path.realpath('../..'))
from alpha_gomoku.cppboard import Board
from alpha_gomoku.datasets import piskvork
from alpha_gomoku.datasets import vct
from alpha_gomoku.tests.test_utils import show_vct
import time


def main():
    path = 'F:/repositories/gomocup/records/gomocup1/0x1-23(1).rec'
    actions = piskvork.load_piskvork_record(path)
    vct_actions = vct.get_vct_actions(actions)
    if len(vct_actions):
        print('vct action found')
        step, _ = vct_actions[0]
        board = Board(actions[:step])
        start = time.time()
        board.evaluate()
        print('vct search spends {:.3f}s'.format(time.time() - start))
        show_vct(board)
    else:
        print('not found vct action')


if __name__ == '__main__':
    main()