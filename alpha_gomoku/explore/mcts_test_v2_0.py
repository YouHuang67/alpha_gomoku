from tqdm import tqdm

from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.explore import value_iteration_v2_0 as vi


def get_model(model='SEWideResnet16_1'):
    weight_dir = utils.DATA_DIR / 'value_iteration' / 'v2' / 'weights'
    weight_paths = sorted(weight_dir.glob(f'{model.lower()}_*.pth'), key=str)
    return vi.get_model(model, weight_paths[0])


class GetMCTS(object):

    def __init__(self, model, **kwargs):
        self.model = model
        self.mcts_args = kwargs

    def __call__(self, boards):
        return vi.MCTS(self.model, boards, **self.mcts_args)


move = lambda xs, acts: list(map(lambda x: x[0].move(x[1]), zip(xs, acts)))


def main():
    root_dir = utils.DATA_DIR / 'mcts_test' / 'v2'
    root_dir.mkdir(parents=True, exist_ok=True)

    model = get_model()

    baseline = GetMCTS(model, visit_times=200, cpuct=1.0, verbose=0,
                       max_node_num=100000)
    mcts_dict = {
        'less_node_num': GetMCTS(model, visit_times=200, cpuct=1.0, verbose=0,
                                 max_node_num=10000),
        'more_cpuct': GetMCTS(model, visit_times=200, cpuct=2.5, verbose=0,
                         max_node_num=100000),
        'more_visit': GetMCTS(model, visit_times=1000, cpuct=1.0, verbose=0,
                              max_node_num=100000)
    }
    boards = [Board(vi.get_random_actions(5)) for _ in range(50)]
    batch_size = 5
    for name, mcts in mcts_dict.items():
        wins = 0.0
        num_samples = 0.0
        with tqdm(range(0, len(boards), batch_size)) as pbar:
            for batch_start in pbar:
                first_wins = []
                second_wins = []
                for first, second in [(baseline, mcts), (mcts, baseline)]:
                    copy_boards = [board.copy() for board in
                                   boards[batch_start:batch_start+batch_size]]
                    first = first(copy_boards)
                    first_players = [board.player for board in copy_boards]

                    actions = first.search()
                    first.move(actions)

                    second = second(move(copy_boards, actions))
                    second_players = [board.player for board in copy_boards]

                    left_boards = copy_boards[:]
                    step = 1
                    while len(left_boards):
                        actions = [first, second][step % 2].search()
                        for board in move(left_boards, actions):
                            if board.is_over:
                                left_boards.remove(board)
                        first.move(actions)
                        second.move(actions)
                        step += 1
                    first_wins.append(sum(int(board.winner == player)
                                          for board, player in
                                          zip(copy_boards, first_players)))
                    second_wins.append(sum(int(board.winner == player)
                                           for board, player in
                                           zip(copy_boards, second_players)))
                wins += first_wins[1] + second_wins[0]
                num_samples += 2 * len(copy_boards)
                pbar.set_description(f'{name}: win {wins / num_samples:.2f}')


if __name__ == '__main__':
    main()