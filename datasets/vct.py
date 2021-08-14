from ..cppboard import Board


def get_vct_actions(actions, max_node_num=100000):
    board = Board()
    vct_actions = []
    for step, act in enumerate(actions):
        action = board.evaluate(max_node_num)
        if board.attacker == board.player:
            vct_actions.append((step, action[0]))
        board.move(act)
    return vct_actions