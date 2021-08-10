import os
import sys
import random

sys.path.append(os.path.realpath('../..'))
from alpha_gomoku.cppboard import Board


def main():
    actions = [112, 97, 98, 113, 96, 95, 81, 66,
			   80, 64, 111, 84, 126, 141, 127, 125]
    actions = [(act // Board.BOARD_SIZE, act % Board.BOARD_SIZE)
                for act in actions]
    board = Board(actions)
    print(board)
    while not board.is_over:
        actions = board.evaluate()
        action = random.choice(actions)
        board.move(action)
        print(board)
        print(action)


if __name__ == '__main__':
    main()