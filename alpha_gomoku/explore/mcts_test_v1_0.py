from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.explore import value_iteration_v2_0 as vi


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


def main():
    root_dir = utils.DATA_DIR / 'value_iteration' / 'v2'
    weight_dir = root_dir / 'weights'
    model = 'SEWideResnet16_1'
    weight_paths = sorted(weight_dir.glob(f'{model.lower()}_*.pth'), key=str)
    model = vi.get_model(model, weight_paths[-1])
    player = get_input('black(0) or white(1) ? ', int, ['0', '1'])
    board = Board()
    if player == 0:
        print(board)
        action = get_board_input(board)
        board.move(action)
    mcts = vi.MCTS(model, [board], visit_times=1000, cpuct=2.5,
                   verbose=1, max_node_num=300000)
    while not board.is_over:
        print(board)
        if board.player != player:
            action = mcts.search()[0]
        root = mcts.node_tables[0].get(board.key, None)
        if root is None:
            print('not found root')
        else:
            children = sorted(root.children.items(),
                              key=lambda x: x[1].visit, reverse=True)
            total_visit = sum(n.visit for act, n in children)
            info = [f'{act}: p {n.prob:.2f} v {n.visit:.0f} s {n.score:.2f} ' +
                    f'b {n.visit_bonus(total_visit):.2f}'
                    for act, n in children[:5]]
            print('\n'.join(info))
        if board.player == player:
            action = get_board_input(board)
        board.move(action)
        mcts.move([action])
    print(board)


if __name__ == '__main__':
    main()