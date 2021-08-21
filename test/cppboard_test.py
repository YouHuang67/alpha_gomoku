from alpha_gomoku.cppboard import Board
from alpha_gomoku.testing import show_vct


def main():
    actions = [112, 97, 98, 113, 96, 95, 81, 66,
			   80, 64, 111, 84, 126, 141, 127, 125]
    actions = [(act // Board.BOARD_SIZE, act % Board.BOARD_SIZE)
                for act in actions]
    board = Board(actions)
    board.evaluate()
    show_vct(board)


if __name__ == '__main__':
    main()