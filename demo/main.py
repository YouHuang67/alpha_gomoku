import argparse

from demo.models import get
from demo.cppboard import Board
from demo.mcts import MonteCarloTreeSearch


def get_player():
    while True:
        ipt = input('player: black (0) or white (1) ? ')
        try:
            player = int(ipt)
            assert player in [0, 1]
            return player
        except:
            print(f'illegal input: {ipt}')


def get_board_input(board):
    while True:
        ipt = input('action: (row id) (col id) like 7 7, '
                    'or undo n steps (-n): ')
        try:
            ipt = ipt.split()
            if len(ipt) == 1:
                return int(ipt[0])
            row, col = tuple(map(int, ipt))
            assert board.is_legal((row, col))
            return (row, col)
        except:
            print(f'illegal action: {ipt}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SEWideResnet16_1', type=str)
    parser.add_argument('--num_iters', default=200, type=int)
    parser.add_argument('--cpuct', default=2.5, type=float)
    parser.add_argument('--max_num_nodes', default=100000, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    mcts = MonteCarloTreeSearch(get(args.model), args.num_iters, args.cpuct,
                                args.max_num_nodes)
    player = get_player()
    board = Board()
    while not board.is_over:
        if board.player == player:
            while True:
                print(board)
                action = get_board_input(board)
                if isinstance(action, tuple):
                    break
                board = Board(board.history[:2 * action])
        else:
            print(board)
            action = mcts.search(board)
            root = mcts.node_table.get(board.key, None)
            if root is None:
                print('not found root')
            else:
                children = sorted(root.children.items(),
                                  key=lambda x: -x[1].visit)
                total_visit = sum(n.visit for act, n in children)
                info = [
                    f'{act}: p {n.prob:.2f} v {n.visit:.0f} s {n.score:.2f} ' +
                    f'b {n.visit_bonus(total_visit):.2f}'
                    for act, n in children[:5]]
                print('\n'.join(info))
        board.move(action)
    print(board)
    print('winner: {}'.format({0: 'black', 1: 'white'}.get(board.winner, 'draw')))


if __name__ == '__main__':
    main()