import random

from alpha_gomoku.cppboard import Board


def show_vct(board):
    assert board.attacker in [Board.BLACK, Board.WHITE]
    print(board)
    while not board.is_over:
        actions = board.evaluate()
        action = random.choice(actions)
        board.move(action)
        print(board)
        print(action)