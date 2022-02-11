import argparse

from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.explore import value_iteration_v3_0 as vi


def get_input(info, dtype, choice):
    while True:
        ipt = input(info)
        if ipt in choice:
            return dtype(ipt)
        print(f'illegal input: {ipt}')


def get_board_input(board):
    size = Board.BOARD_SIZE
    legal_actions = []
    for act, value in enumerate(board.vector):
        if value != 2:
            continue
        legal_actions.append(f'{act // size} {act % size}')
    return get_input('get action: ',
                     lambda s: tuple(map(int, s.split())),
                     legal_actions)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SEWideResnet16_1', type=str)
    parser.add_argument('--index', default=0, type=int)
    # mcts setting
    parser.add_argument('--num_iters', default=200, type=int)
    parser.add_argument('--cpuct', default=2.5, type=float)
    parser.add_argument('--max_node_num', default=100000, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = utils.DATA_DIR / 'value_iteration' / 'v3'
    weight_dir = root_dir / 'weights'
    weight_path = weight_dir / f'{args.model.lower()}_{args.index:02d}.pth'
    model = vi.get_model(args.model, weight_path)
    kwargs = utils.get_func_kwargs(vi.MonteCarloTreeSearch.__init__,
                                   args.__dict__)
    kwargs.pop('model')
    mcts = vi.MonteCarloTreeSearch(model, perturb_root=True, **kwargs)
    player = get_input('black(0) or white(1) ? ', int, ['0', '1'])
    board = Board()
    while not board.is_over:
        print(board)
        if board.player == player:
            action = get_board_input(board)
        else:
            action = mcts.search([board], taus=0.0)[0]
            root = mcts.node_tables[board].get(board.key, None)
            if root is None:
                print('not found root')
            else:
                children = sorted(root.children.items(), key=lambda x: -x[1].visit)
                total_visit = sum(n.visit for act, n in children)
                info = [f'{act}: p {n.prob:.2f} v {n.visit:.0f} s {n.score:.2f} ' +
                        f'b {n.visit_bonus(total_visit):.2f}'
                        for act, n in children[:5]]
                print('\n'.join(info))
        board.move(action)
    print(board)
    print('winner: {}'.format({0: 'black', 1: 'white'}.get(board.winner, 'draw')))


if __name__ == '__main__':
    main()