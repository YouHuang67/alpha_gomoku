import argparse
from tqdm import tqdm

import pytorch_lightning as pl

from alpha_gomoku import utils
from alpha_gomoku.explore import value_iteration_v3_0 as vi


def self_play(mcts_1, mcts_2, boards, batch_size):
    winners = []
    for batch_start in tqdm(range(0, len(boards), batch_size)):
        batch_boards = [boards[i].copy() for i in
                        range(batch_start, batch_start + batch_size)
                        if i < len(boards)]
        left_boards = batch_boards[:]
        pla_1, pla_2 = mcts_1, mcts_2
        while len(left_boards):
            for board, action in list(zip(left_boards,
                                          pla_1.search(left_boards, taus=0.0))):
                if board.move(action).is_over:
                    left_boards.remove(board)
            pla_1, pla_2 = pla_2, pla_1
        winners.extend([board.winner for board in batch_boards])
        mcts_1.reset()
        mcts_2.reset()
    wins_1 = sum(int(winner == board.player)
                 for winner, board in zip(winners, boards))
    wins_2 = sum(int(winner == (1 - board.player))
                 for winner, board in zip(winners, boards))
    return wins_1, wins_2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_1', type=str, required=True)
    parser.add_argument('--model_2', type=str, required=True)
    parser.add_argument('--path_1', type=str, required=True)
    parser.add_argument('--path_2', type=str, required=True)
    parser.add_argument('--num_iters', default=200, type=int)
    parser.add_argument('--max_node_num', default=100000, type=int)
    parser.add_argument('--num_boards', default=50, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    args.verbose = 0
    args.perturb = False
    pl.seed_everything(args.seed)
    boards = vi.BoardGenerator()(args.num_boards)
    mcts_1 = vi.MonteCarloTreeSearch(vi.get_model(args.model_1,
                                                  utils.ROOT / args.path_1),
                                     **args.__dict__)
    mcts_2 = vi.MonteCarloTreeSearch(vi.get_model(args.model_2,
                                                  utils.ROOT / args.path_2),
                                     **args.__dict__)
    wins_1, wins_2 = 0, 0
    w1, w2 = self_play(mcts_1, mcts_2, boards, args.batch_size)
    wins_1 += w1
    wins_2 += w2
    w2, w1 = self_play(mcts_2, mcts_1, boards, args.batch_size)
    wins_1 += w1
    wins_2 += w2
    win_ratio_1 = wins_1 / float(2 * args.num_boards)
    win_ratio_2 = wins_2 / float(2 * args.num_boards)
    print(f'win 1: {win_ratio_1:.4f} win 2: {win_ratio_2:.4f}')


if __name__ == '__main__':
    main()
