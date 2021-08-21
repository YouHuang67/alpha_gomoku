import time
import random
from pathlib import Path

from alpha_gomoku.cppboard import Board
from alpha_gomoku.datasets import piskvork
from alpha_gomoku.datasets import vct
from alpha_gomoku.testing import show_vct


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


def test_piskvork_vct_actions(root):
    vct_action_dataset = piskvork.PiskvorkVCTActions(root)
    print(f'found {len(vct_action_dataset)} vct actions')
    vct_action_dataset.save(Path(root) / 'vct_actions.json')
    vct_action_dataset = piskvork.PiskvorkVCTActions()
    vct_action_dataset.load(Path(root) / 'vct_actions.json')
    index = random.randint(0, len(vct_action_dataset) - 1)
    actions, action = vct_action_dataset[index]
    print(Board(actions))
    print(action)


def main():
    # test_single_board('F:/repositories/gomocup/records/gomocup1/0x1-23(1).rec')
    # count_boards('F:/repositories/gomocup/records/')
    test_piskvork_vct_actions('F:/repositories/gomocup/records/')


if __name__ == '__main__':
    main()