from ..cppboard import Board


def get_vct_actions(actions, max_node_num=100000):
    board = Board()
    vct_actions = []
    for step, act in enumerate(actions):
        action = board.evaluate(max_node_num)
        if board.attacker in [Board.BLACK, Board.WHITE] and \
                board.attacker == board.player:
            if not board.is_legal(action[0]):
                print(board)
                print(f'Illegal action: {action}')
                print(f'Evaluate Board: {Board(actions[:step]).evaluate()}')
                print('previous actions: ')
                print(actions[:step])
                exit(0)
            vct_actions.append((step, action[0]))
        board.move(act)
    return vct_actions