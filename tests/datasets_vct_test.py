import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from alpha_gomoku.cppboard import Board
from alpha_gomoku.datasets import piskvork
from alpha_gomoku.datasets import vct
from alpha_gomoku.tests.test_utils import show_vct


def test_single_board(path):
    actions = piskvork.load_piskvork_record(path)
    vct_actions = vct.get_vct_actions(actions)
    if len(vct_actions):
        step, _ = vct_actions[0]
        print('vct action found at step {}'.format(step))
        board = Board(actions[:step])
        start = time.time()
        board.evaluate()
        print('vct search spends {:.3f}s'.format(time.time() - start))
        show_vct(board)
    else:
        print('not found vct action')


def count_boards(root):
    keys = []
    for actions in piskvork.load_piskvork_records(root):
        boards = [Board() for _ in range(8)]
        for action in actions:
            for board, act in zip(boards, Board.get_homogenous_actions(action)):
                board.move(act)
        keys.extend([board.key for board in boards])
    print(f'found {len(keys) // 8} boards')
    print(f'{len(set(keys)) // 8} different boards')


def main():
    # test_single_board('F:/repositories/gomocup/records/gomocup1/0x1-23(1).rec')
    count_boards('F:/repositories/gomocup/records/')


if __name__ == '__main__':
    main()