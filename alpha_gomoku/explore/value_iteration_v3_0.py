import gc
import yaml
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.explore import base
from alpha_gomoku.explore import models
from alpha_gomoku.explore import schedule
from alpha_gomoku.mcts.mcts_v2 import Node


SIZE = Board.BOARD_SIZE
NUM_STONES = SIZE ** 2
DEVICE = utils.DEVICE
AUGMENTATION_FUNCS = list(utils.AUGMENTATION_FUNCS.values())
INVERSE_FUNCS = list(utils.INVERSE_FUNCS.values())
NUM_FUNCS = len(AUGMENTATION_FUNCS)
ROOT_DIR = utils.DATA_DIR / 'value_iteration' / 'v3'
WEIGHT_DIR = ROOT_DIR / 'weights'
HISTORY_DIR = ROOT_DIR / 'history'


def flatten(*args):
    row, col = (args[0] if len(args) == 1 else args)
    return row * SIZE + col


def unflatten(action):
    return action // SIZE, action % SIZE


class Vicinity(nn.Module):

    def __init__(self, radius=3):
        super(Vicinity, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(radius * 2 + 1),
                              stride=1, padding=radius, bias=False)
        nn.init.ones_(self.conv.weight)
        self.conv.weight.requires_grad_(False)

    @torch.no_grad()
    def forward(self, masks):
        shape = masks.size()
        masks = masks.view(-1, 1, SIZE, SIZE).float()
        zero_masks = (masks.long().reshape(-1, NUM_STONES).sum(-1) == 0)
        if zero_masks.any():
            masks[zero_masks, 0, SIZE // 2, SIZE // 2] = 1.0
        return (self.conv(masks) > 0.0).view(*shape).detach().to(masks.device)


class NetWrapper(nn.Module):

    def __init__(self, model, dihedral_reflection='rand', gap=3, **kwargs):
        assert dihedral_reflection in ['rand', 'mean', 'none']
        super(NetWrapper, self).__init__()
        self.model = model
        self.dihedral_reflection = dihedral_reflection
        self.vicinity = Vicinity(gap)

    @staticmethod
    def actions_to_tensors(action_list):
        tensors = []
        size = torch.Size([SIZE, SIZE])
        for actions in action_list:
            indices = torch.LongTensor(actions).transpose(0, 1)
            values = torch.ones(len(actions)).long()
            tensor = torch.sparse_coo_tensor(indices, values, size)
            tensors.append(tensor.to_dense())
        return torch.stack(tensors, dim=0).to(DEVICE)

    @staticmethod
    def boards_to_tensors(board_list):
        return torch.stack([torch.LongTensor(board.vector).view(SIZE, SIZE)
                            for board in board_list], dim=0).to(DEVICE), \
               torch.LongTensor([board.player
                                 for board in board_list]).to(DEVICE)

    @torch.no_grad()
    def forward(self, board_list):
        board_tensors, players = self.boards_to_tensors(board_list)
        legal_masks = (board_tensors == 2)
        if self.dihedral_reflection == 'mean':
            board_tensors = torch.cat([func(board_tensors) for func in
                                       AUGMENTATION_FUNCS], dim=0)
            players = players.repeat(NUM_FUNCS)
            logits, values = self.model((board_tensors, players))
            logits = logits.view(NUM_FUNCS, -1, SIZE, SIZE)
            logits = torch.stack([func(logit) for func, logit in
                                  zip(INVERSE_FUNCS, logits)], dim=0).mean(0)
            values = values.view(NUM_FUNCS, -1).mean(0)
        else:
            index = random.randint(0, len(utils.AUGMENTATION_FUNCS) - 1) \
                    if self.dihedral_reflection == 'rand' else 0
            func, inv_func = AUGMENTATION_FUNCS[index], INVERSE_FUNCS[index]
            logits, values = self.model((func(board_tensors), players))
            logits = inv_func(logits.view(-1, SIZE, SIZE))

        logits = logits.detach()
        logits[~legal_masks] = -torch.inf
        probs = F.softmax(logits.reshape(-1, NUM_STONES), -1)

        masks = legal_masks & self.vicinity(~legal_masks)
        for index, board in enumerate(board_list):
            actions = board.copy().evaluate(1)
            if len(actions):
                masks[index, ~self.actions_to_tensors([actions])[0].bool()] = False
        masks = masks.view(-1, NUM_STONES)
        probs /= (masks.float() * probs).sum(-1, keepdim=True)

        results = []
        for i, (ps, val) in enumerate(zip(probs, values)):
            results.append((dict(), val.item()))
            for act in torch.where(masks[i])[0].cpu().numpy().tolist():
                results[-1][0][unflatten(act)] = ps[act].item()
        return results


class EnsembleOutput(nn.Module):

    def __init__(self):
        super(EnsembleOutput, self).__init__()
        self.bn = nn.BatchNorm1d(Board.BOARD_SIZE ** 2)
        self.fc = nn.Linear(Board.BOARD_SIZE ** 2, 1, bias=True)

    def forward(self, x):
        out = self.bn(x.view(x.size(0), -1))
        out = self.fc(out).view(-1)
        out = torch.tanh(out)
        return x, out


def get_model(backbone, weight_path=''):
    backbone = backbone.lower()
    backbone = {k.lower(): v for k, v in models.__dict__.items()}[backbone]()
    model = nn.Sequential(models.BoardToTensor(), backbone, EnsembleOutput())
    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    return model.eval().to(DEVICE)


class MonteCarloTreeSearch(object):

    def __init__(self, model, num_iters=200, cpuct=2.5,
                 max_node_num=100000, verbose=1, noise_alpha=0.03,
                 noise_gamma=0.25, perturb_root=True, **kwargs):
        self.model = NetWrapper(model, **kwargs).to(DEVICE)
        self.num_iters, self.cpuct = num_iters, cpuct
        self.max_node_num, self.verbose = max_node_num, verbose
        self.noise_alpha, self.noise_gamma = noise_alpha, noise_gamma
        self.node_tables = OrderedDict()
        self.visit_tables = OrderedDict()
        self.perturb_root = perturb_root

    def perturb(self, root):
        if len(root.children) == 0 or self.noise_gamma == 0.0:
            return
        noise = np.random.dirichlet([self.noise_alpha] * len(root.children))
        gamma = self.noise_gamma
        for index, node in enumerate(root.children.values()):
            node._prob = (1 - gamma) * node.prob + gamma * noise[index]
            node._prob_n = 1

    def evaluate(self, boards, description=''):
        roots = []
        indices = []
        for index, board in enumerate(boards):
            copy_board = board.copy()
            actions = copy_board.evaluate(self.max_node_num)
            if len(actions) and copy_board.attacker == copy_board.player:
                visit_table = self.visit_tables.setdefault(board, dict())
                visit_table[board.key] = {actions[0]: 1}
            else:
                node_table = self.node_tables.setdefault(board, dict())
                root = Node.get_node(board, node_table,
                                     cpuct=self.cpuct, table=node_table)
                root.board = board
                roots.append(root)
                indices.append(index)

        if len(roots) == 0:
            return [self.visit_tables[board][board.key] for board in boards]

        iters = range(self.num_iters)
        iters = tqdm(iters, description) if self.verbose else iters
        perturb_flags = {root: self.perturb_root for root in roots}
        for _ in iters:
            nodes = []
            depths = []
            leaf_boards = []
            for root in roots:
                board = root.board.copy()
                node, depth = root.forward(board)

                max_node_num = self.max_node_num
                if root not in list(node.parents.values()) + [node]:
                    max_node_num //= 10
                if node.is_leaf or node.evaluate_leaf(board.copy(), max_node_num):
                    node.backward(node.leaf_value, depth)
                    continue

                nodes.append(node)
                depths.append(depth)
                leaf_boards.append(board)

            if len(leaf_boards) == 0:
                continue

            for index, (probs, value) in enumerate(self.model(leaf_boards)):
                node, depth = nodes[index], depths[index]
                node.expand(probs, leaf_boards[index], node.table)
                node.backward(value, depth)

            for root in roots:
                if perturb_flags[root]:
                    self.perturb(root)
                    perturb_flags[root] = False

        for index, root in zip(indices, roots):
            board = boards[index]
            visit_table = self.visit_tables.setdefault(board, dict())
            visit_table[board.key] = {act: int(node.visit) for act, node in
                                      root.children.items()}

        return [self.visit_tables[board][board.key] for board in boards]

    @staticmethod
    def sample_actions(visit_tables, taus=1.0):
        if isinstance(taus, (int, float)):
            taus = [taus] * len(visit_tables)
        final_actions = []
        for visit_table, tau in zip(visit_tables, taus):
            actions, visits = list(zip(*list(visit_table.items())))
            if tau == 0.0:
                action = actions[np.argmax(visits)]
            else:
                visits = [float(v) ** (1.0 / tau) for v in visits]
                total_visits = sum(visits)
                probs = [v / total_visits for v in visits]
                rand, s = random.random(), 0.0
                for i, p in enumerate(probs):
                    if s <= rand < s + p:
                        action = actions[i]
                        break
                    s += p
            final_actions.append(action)
        return final_actions

    def search(self, boards, description='', taus=1.0):
        return self.sample_actions(self.evaluate(boards, description), taus)

    def reset(self):
        self.node_tables.clear()
        self.visit_tables.clear()
        gc.collect()


class BoardGenerator(object):

    def __init__(self):
        dataset, vct_dataset = OrderedDict(), base.VCTActions().dataset
        for index, step, _ in vct_dataset.vct_actions:
            dataset[index] = min(step, dataset.get(index, NUM_STONES))
        self.dataset = dataset
        self.actions = vct_dataset.actions

    def __call__(self, num_boards):
        dataset = list(self.dataset.items())
        random.shuffle(dataset)
        boards = []
        for index, step in dataset[:num_boards]:
            step = random.randint(0, step)
            actions = self.actions[index][:step]
            actions = list(zip(*[Board.get_homogenous_actions(act)
                                 for act in actions]))
            if len(actions):
                boards.append(Board(random.choice(actions)))
            else:
                boards.append(Board())
        return boards


BOARD_GENERATOR = BoardGenerator()


class SelfPlayPipeline(object):

    def __init__(self, backbone, num_boards, batch_size,
                 num_iters=200, cpuct=2.5, max_node_num=100000,
                 noise_alpha=0.03, noise_gamma=0.25, gap=3, **kwargs):
        backbone = backbone.lower()
        self.num_boards = num_boards
        self.batch_size = batch_size
        kwargs = {'num_iters': num_iters, 'cpuct': cpuct, 'verbose': 0,
                  'max_node_num': max_node_num, 'noise_alpha': noise_alpha,
                  'noise_gamma': noise_gamma, 'gap': gap}
        weight_paths = sorted(WEIGHT_DIR.glob(f'{backbone}_*.pth'), key=str)
        self.mcts_list = [MonteCarloTreeSearch(get_model(backbone, weight_path), **kwargs)
                          for weight_path in weight_paths]

    def self_play(self, mcts_1, mcts_2, boards, description='', taus=1.0):
        batch_size = self.batch_size
        history_list = [board.history[:] for board in boards]
        players = [board.player for board in boards]
        records = OrderedDict()
        for batch_start in tqdm(range(0, len(boards), batch_size), description):
            batch_boards = boards[batch_start:batch_start + batch_size]
            pla_1, pla_2 = mcts_1, mcts_2
            while len(batch_boards):
                visit_tables = pla_1.evaluate(batch_boards)
                actions = pla_1.sample_actions(visit_tables, taus=taus)
                for index, board in list(enumerate(batch_boards)):
                    visit_table = [(act, v) for act, v in
                                   visit_tables[index].items() if v > 0]
                    record = (visit_table, actions[index])
                    records.setdefault(board, list()).append(record)
                    if board.move(actions[index]).is_over:
                        batch_boards.remove(board)
                pla_1, pla_2 = pla_2, pla_1
            mcts_1.reset()
            mcts_2.reset()
        samples, wins_1, wins_2 = [], 0.0, 0.0
        for index, board in enumerate(boards):
            samples.append((history_list[index], records[board]))
            wins_1 += float(board.winner == players[index])
            wins_2 += float(board.winner == (1 - players[index]))
        return samples, wins_1 / len(boards), wins_2 / len(boards)

    def evaluate(self, index_1, index_2):
        boards = BOARD_GENERATOR(50)
        win_ratio_1, win_ratio_2 = 0.0, 0.0
        mcts_1 = self.mcts_list[index_1]
        mcts_2 = self.mcts_list[index_2]
        mcts_1.perturb_root = False
        mcts_2.perturb_root = False
        _, ws1, ws2 = self.self_play(mcts_1, mcts_2,
                                     [board.copy() for board in boards],
                                     f'{index_1} vs {index_2}', 0.0)
        win_ratio_1 += ws1
        win_ratio_2 += ws2
        _, ws2, ws1 = self.self_play(mcts_2, mcts_1,
                                     [board.copy() for board in boards],
                                     f'{index_2} vs {index_1}', 0.0)
        win_ratio_1 += ws1
        win_ratio_2 += ws2
        return win_ratio_1 / 2, win_ratio_2 / 2

    def get_best(self):
        mcts_list = self.mcts_list
        result_path = WEIGHT_DIR / 'results.json'
        if not result_path.is_file():
            results = {'best': 0, 'results': [[None]]}
            utils.json_save(result_path, results)
        results = utils.json_load(result_path)
        if len(results['results']) == len(mcts_list):
            return results['best']
        for index in range(len(mcts_list)):
            if index < len(results['results']):
                continue
            results['results'].append(list())
            for target_index in range(index):
                win_ratio_1, win_ratio_2 = self.evaluate(index, target_index)
                results['results'][index].append(win_ratio_1)
                results['results'][target_index].append(win_ratio_2)
            results['results'][index].append(None)
            best_index = results['best']
            if results['results'][best_index][index] < \
                    results['results'][index][best_index]:
                results['best'] = index
        utils.json_save(result_path, results)
        return results['best']

    def run(self, description=''):
        best = self.get_best()
        best_mcts = self.mcts_list[best]
        best_mcts.perturb_root = True
        history_paths = sorted(HISTORY_DIR.glob('*.json'), key=str)
        history_path = HISTORY_DIR / f'{len(history_paths):04d}.json'
        if description:
            description = f'{description}-{best}-{len(history_paths)}'
        else:
            description = f'{best}-{len(history_paths)}'
        start = time.time()
        samples, *_ = self.self_play(best_mcts, best_mcts,
                                     BOARD_GENERATOR(self.num_boards),
                                     description, taus=1.0)
        total_time = time.time() - start
        history_path.parents[0].mkdir(parents=True, exist_ok=True)
        utils.json_save(history_path, samples)
        num_samples = sum(len(records) for _, records in samples)
        print(f'{num_samples} samples store, ' +
              f'with each spending {total_time / num_samples:.4f}s')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, container_size, tau=1.0):
        self.history_list = []
        self.quadruples = []
        for history, records in self.sample_generator():
            history = list(map(tuple, history))
            actions = [tuple(record[1]) for record in records]
            winner = Board(history + actions).winner
            index = len(self.history_list)
            self.history_list.append(history + actions)
            for step, (visit_table, _) in enumerate(records, len(history)):
                actions, visits = [], []
                for act, v in visit_table:
                    actions.append(act)
                    visits.append(v ** (1.0 / tau))
                total_visit = sum(visits)
                visits = [v / total_visit for v in visits]
                value = (float(winner == (step % 2)) - 0.5) * 2
                self.quadruples.append((index, step, (actions, visits), value))
            if len(self.quadruples) >= container_size:
                break

    @staticmethod
    def sample_generator():
        for path in sorted(HISTORY_DIR.glob('*.json'), key=lambda x: -int(x.stem)):
            samples = utils.json_load(path)
            for sample in samples:
                yield sample

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, item):
        index, step, (actions, visits), value = self.quadruples[item]
        previous_actions = self.history_list[index][:step]
        size = torch.Size([SIZE, SIZE])

        if step:
            indices = torch.LongTensor(list(zip(*previous_actions)))
            values = torch.LongTensor([(i % 2) - 2 for i in range(len(previous_actions))])
            board_tensor = torch.sparse.LongTensor(indices, values, size).to_dense() + 2
        else:
            board_tensor = torch.zeros(SIZE, SIZE).long() + 2
        player = step % 2

        indices = torch.LongTensor(list(zip(*actions)))
        values = torch.FloatTensor(visits)
        action_tensor = torch.sparse_coo_tensor(indices, values, size).to_dense()

        func = random.choice(list(AUGMENTATION_FUNCS))
        board_tensor = func(board_tensor.unsqueeze(0)).squeeze(0)
        action_tensor = func(action_tensor.unsqueeze(0)).reshape(-1)

        return board_tensor, player, action_tensor, value


class PolicyPipeline(pl.LightningModule):

    def __init__(self, **kwargs):
        super(PolicyPipeline, self).__init__()
        self.save_hyperparameters()
        args = self.hparams
        if args.from_scratch:
            weight_path = ''
        else:
            weight_path = WEIGHT_DIR / f'{args.model.lower()}_{args.model_index:02d}.pth'
        self.model = get_model(args.model, weight_path)
        self.dataset = Dataset(args.container_size)

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr_max,
            momentum=0.9, weight_decay=5e-4
        )
        return [optimizer], [schedule.get_scheduler(args, optimizer)]

    def training_step(self, batch, batch_idx):
        *inputs, action_tensors, values = batch
        pred_log, pred_val = self.model(inputs)
        losses = []

        legal_masks = (inputs[0] == 2).reshape(*pred_log.size())
        neg_inf = torch.zeros_like(pred_log) - torch.inf
        pred_log = torch.where(legal_masks, pred_log, neg_inf)
        log_prob = F.log_softmax(pred_log, -1)
        log_prob = torch.where(legal_masks, log_prob, torch.zeros_like(log_prob))
        policy_loss = -(action_tensors * log_prob).sum(-1).mean()
        self.log('policy_loss', policy_loss, on_step=False, on_epoch=True)
        losses.append(policy_loss)

        value_loss = (pred_val - values).square().mean()
        self.log('value_loss', value_loss, on_step=False, on_epoch=True)
        losses.append(value_loss)

        return sum(losses)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=128,
            shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return self.train_dataloader()

    def on_fit_end(self):
        args = self.hparams
        index = args.next_model_index
        weight_path = WEIGHT_DIR / f'{args.model.lower()}_{index:02d}.pth'
        torch.save(self.model.state_dict(), weight_path)


def parse_args():
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='SEWideResnet16_1', type=str)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr_schedule', default='cosine',
                        choices=['superconverge', 'piecewise', 'linear',
                                 'piecewisesmoothed', 'piecewisezoom',
                                 'onedrop', 'multipledecay', 'cosine', 'none'])
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--lr_one_drop', default=0.01, type=float)
    parser.add_argument('--lr_drop_epoch', default=100, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    # self play
    parser.add_argument('--num_boards', default=100, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--container_size', default=100000, type=int)
    parser.add_argument('--num_iters', default=200, type=int)
    parser.add_argument('--cpuct', default=2.5, type=float)
    parser.add_argument('--max_node_num', default=100000, type=int)
    parser.add_argument('--rand_init', action='store_true')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--num_runs', default=10, type=int)
    parser.add_argument('--noise_alpha', default=0.03, type=float)
    parser.add_argument('--noise_gamma', default=0.25, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    # pl.seed_everything(args.seed)
    if not WEIGHT_DIR.is_dir():
        WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
        if args.rand_init:
            trained_weight_path = ''
        else:
            trained_weight_path = utils.DATA_DIR / \
                                  f'weights/explore/ensemble/v1/{args.model.lower()}.pth'
        init_model = get_model(args.model, trained_weight_path)
        init_weight_path = WEIGHT_DIR / f'{args.model.lower()}_{0:02d}.pth'
        torch.save(init_model.state_dict(), init_weight_path)

    gpus = 1 if torch.cuda.is_available() else 0
    accelerator = 'dp' if torch.cuda.is_available() else 'cpu'

    while True:
        weight_paths = list(WEIGHT_DIR.glob(f'{args.model.lower()}_*.pth'))
        num_weights = len(weight_paths)

        num_runs = len(list(HISTORY_DIR.glob('*.json')))
        target_num_runs = num_weights * args.num_runs
        sp = SelfPlayPipeline(args.model, **args.__dict__)
        for index in range(num_runs, target_num_runs):
            sp.run()

        ppl = PolicyPipeline(**args.__dict__, version='3.0',
                             model_index=sp.get_best(),
                             next_model_index=num_weights)
        trainer = pl.Trainer(gpus=gpus, accelerator=accelerator,
                             max_epochs=ppl.hparams.epochs,
                             default_root_dir=ROOT_DIR)
        trainer.fit(ppl)


if __name__ == '__main__':
    main()
