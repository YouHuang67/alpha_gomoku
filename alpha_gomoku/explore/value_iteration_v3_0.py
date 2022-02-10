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
            for act in torch.where(masks[i])[0].numpy().tolist():
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

    def __init__(self, model, num_iters=200, cpuct=2.5, max_node_num=100000,
                 verbose=1, noise_alpha=0.03, noise_gamma=0.25, **kwargs):
        self.model = NetWrapper(model, **kwargs)
        self.num_iters, self.cpuct = num_iters, cpuct
        self.max_node_num, self.verbose = max_node_num, verbose
        self.noise_alpha, self.noise_gamma = noise_alpha, noise_gamma
        self.node_tables = OrderedDict()
        self.visit_tables = OrderedDict()

    def perturb(self, root):
        if len(root.children) == 0:
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
        perturb_flags = {root: True for root in roots}
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
            boards.append(Board(random.choice(actions)))
        return boards


BOARD_GENERATOR = BoardGenerator()


class SelfPlayPipeline(object):

    def __init__(self, backbone, num_boards, batch_size,
                 num_iters=200, cpuct=2.5, max_node_num=100000,
                 noise_alpha=0.03, noise_gamma=0.25, gap=3):
        backbone = backbone.lower()
        self.num_boards = num_boards
        self.batch_size = batch_size
        kwargs = {'num_iters': num_iters, 'cpuct': cpuct, 'verbose': 1,
                  'max_node_num': max_node_num, 'noise_alpha': noise_alpha,
                  'noise_gamma': noise_gamma, 'gap': gap}
        weight_paths = sorted(WEIGHT_DIR.glob(f'{backbone}_*.pth'), key=str)
        self.mcts_list = [MonteCarloTreeSearch(get_model(backbone, weight_path), **kwargs)
                          for weight_path in weight_paths]

    def self_play(self, mcts_1, mcts_2, boards, description=''):
        batch_size = self.batch_size
        history_list = [board.history[:] for board in boards]
        players = [board.player for board in boards]
        records = OrderedDict()
        for batch_start in range(0, len(boards), batch_size):
            batch_boards = boards[batch_start:batch_start + batch_size]
            pla_1, pla_2 = mcts_1, mcts_2
            while len(batch_boards):
                visit_tables = pla_1.evaluate(batch_boards, description)
                actions = pla_1.sample_actions(visit_tables, taus=1.0)
                for index, board in list(enumerate(batch_boards)):
                    record = (visit_tables[index], actions[index])
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


pl.seed_everything(100)
sp = SelfPlayPipeline('SEWideResnet16_1', 10, 5)
sp.self_play(sp.mcts_list[0], sp.mcts_list[0], BOARD_GENERATOR(10))


def get_random_actions(steps, gap=3):
    if steps == 0:
        return []
    size = Board.BOARD_SIZE
    actions = torch.rand(size ** 2).argsort().numpy().tolist()
    random_actions = [(actions[0] // size, actions[0] % size)]
    for action in actions[1:]:
        if len(random_actions) == steps:
            break
        row, col = action // size, action % size
        if any([abs(row - r) <= gap and abs(col - c) <= gap
                for r, c in random_actions]):
            random_actions.append((row, col))
    return random_actions


def self_play(model_1, model_2, num_boards, visit_times=200, cpuct=1.0, verbose=1,
              max_node_num=100000, max_random_steps=5, prefix=''):
    move = lambda bds, acts: list(map(lambda x: x[0].move(x[1]), list(zip(bds, acts))))
    gc.collect()
    boards = [Board(get_random_actions(random.randint(0, max_random_steps)))
              for _ in range(num_boards)]
    players = [board.player for board in boards]

    mcts_1 = MCTS(model_1, boards, visit_times, cpuct,
                  verbose, max_node_num)
    actions = mcts_1.search(f'{prefix} step: 0')
    move(boards, actions)
    mcts_1.move(actions)

    mcts_2 = MCTS(model_2, boards, visit_times, cpuct,
                  verbose, max_node_num)

    left_boards = boards[:]
    for step in range(1, 225):
        player = [mcts_1, mcts_2][step % 2]
        actions = player.search(f'{prefix} step: {step:d}')
        mcts_1.move(actions)
        mcts_2.move(actions)
        move(left_boards, actions)
        left_boards = [board for board in left_boards if not board.is_over]
        if len(left_boards) == 0:
            break

    return boards, sum(int(board.winner == player)
                       for board, player in zip(boards, players))


class SelfPlayPipeline(object):

    def __init__(self, root_dir, model, num_boards, visit_times=200, cpuct=1.0,
                 verbose=1, max_node_num=100000, max_random_steps=5, **kwargs):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        self.root_dir = Path(root_dir)
        weight_dir = self.root_dir / 'weights'
        weight_paths = sorted(weight_dir.glob(model.lower() + '*.pth'), key=str)
        self.main_net = get_model(model, weight_paths[-1])
        self.nets = [get_model(model, path) for path in weight_paths]

    def run(self, prefix=''):
        main_net = self.main_net
        history_list = []
        start = time.time()
        results = []
        for index, net in enumerate(self.nets):
            main_wins = 0
            boards, wins = self_play(main_net, net, self.num_boards,
                                     self.visit_times, self.cpuct,
                                     self.verbose, self.max_node_num,
                                     max_random_steps=self.max_random_steps,
                                     prefix=f'{prefix} main vs net_{index}')
            gc.collect()
            history_list.extend([board.history for board in boards])
            main_wins += wins

            boards, wins = self_play(net, main_net, self.num_boards,
                                     self.visit_times, self.cpuct,
                                     self.verbose, self.max_node_num,
                                     max_random_steps=self.max_random_steps,
                                     prefix=f'{prefix} net_{index} vs main')
            gc.collect()
            history_list.extend([board.history for board in boards])
            main_wins += len(boards) - wins

            ratio = main_wins / float(2 * len(boards))
            results.append(f'net {index}: {ratio:.4f}')

        history_dir = self.root_dir / 'history'
        history_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(history_dir.glob('*.json')))
        utils.json_save(history_dir / f'{index:04d}.json', history_list)
        num_actions = sum(len(his) for his in history_list)
        print(f'{num_actions} actions stored, ' +
              f'each action spends {(time.time() - start) / num_actions:.4f}')
        results = ' | '.join(results)
        print(results)
        with open(history_dir / 'results.txt', 'a') as file:
            file.write(f'{prefix} {results}\n')


class Dataset(torch.utils.data.Dataset):
    size = Board.BOARD_SIZE
    num_stones = Board.BOARD_SIZE ** 2

    def __init__(self, history_list, augmentation):
        self.history_list = history_list
        self.augmentation = augmentation
        self.triples = []
        for index, history in enumerate(history_list):
            winner = Board(history).winner
            for step in range(len(history)):
                if winner not in [0, 1]:
                    value = 0.0
                else:
                    value = 1.0 if (step % 2) == winner else -1.0
                self.triples.append((index, step, value))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, item):
        index, step, value = self.triples[item]
        previous_actions = self.history_list[index][:step]

        if step:
            indices = torch.LongTensor(list(zip(*previous_actions)))
            values = torch.LongTensor([(i % 2) - 2 for i in range(len(previous_actions))])
            size = torch.Size([self.size, self.size])
            board_tensor = torch.sparse.LongTensor(indices, values, size).to_dense() + 2
        else:
            board_tensor = torch.zeros(self.size, self.size).long() + 2
        player = step % 2
        action = MCTS.flatten(self.history_list[index][step])

        if self.augmentation:
            func = random.choice(list(utils.AUGMENTATION_FUNCS.values()))
            board_tensor = func(board_tensor.unsqueeze(0)).squeeze(0)
            action_tensor = F.one_hot(torch.LongTensor([action]), self.num_stones)
            action_tensor = func(action_tensor.view(1, self.size, self.size))
            action = torch.argmax(action_tensor.reshape(-1)).item()

        return board_tensor, player, action, value


class PolicyPipeline(pl.LightningModule):

    def __init__(self, **kwargs):
        super(PolicyPipeline, self).__init__()
        self.save_hyperparameters()
        args = self.hparams

        self.root_dir = utils.DATA_DIR / 'value_iteration' / 'v3'
        self.data_dir = self.root_dir / 'history'
        self.weight_dir = self.root_dir / 'weights'
        weight_path = self.weight_dir / f'{args.model.lower()}_{args.model_index:02d}.pth'
        self.model = get_model(args.model, weight_path)

        self.dataset = self.make_dataset(args.container_size)

    @staticmethod
    def make_dataset(container_size):
        history_list = []
        root_dir = utils.DATA_DIR / 'value_iteration' / 'v3'
        data_dir = root_dir / 'history'
        for path in sorted(data_dir.glob('*.json'),
                           key=str, reverse=True)[:container_size]:
            history_list += [list(map(tuple, acts)) for acts in
                             utils.json_load(path)]
        return Dataset(history_list, True)

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr_max,
            momentum=0.9, weight_decay=5e-4
        )
        return [optimizer], [schedule.get_scheduler(args, optimizer)]

    def training_step(self, batch, batch_idx):
        *inputs, actions, values = batch
        pred_log, pred_val = self.model(inputs)
        losses = []

        policy_loss = F.cross_entropy(pred_log, actions)
        self.log('policy_loss', policy_loss, on_step=False, on_epoch=True)
        losses.append(policy_loss)

        value_loss = (pred_val - values).square().mean()
        self.log('value_loss', value_loss, on_step=False, on_epoch=True)
        losses.append(value_loss)

        return sum(losses)

    def train_dataloader(self):
        args = self.hparams
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=2, drop_last=True
        )

    def val_dataloader(self):
        return self.train_dataloader()

    def on_fit_end(self):
        args = self.hparams
        index = args.model_index + 1
        weight_path = self.weight_dir / f'{args.model.lower()}_{index:02d}.pth'
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
    parser.add_argument('--batch_size', default=128, type=int)
    # self play
    parser.add_argument('--num_boards', default=5, type=int)
    parser.add_argument('--container_size', default=20, type=int)
    parser.add_argument('--num_kept_models', default=5, type=int)
    parser.add_argument('--include_policy', action='store_true')
    parser.add_argument('--visit_times', default=200, type=int)
    parser.add_argument('--cpuct', default=1.0, type=float)
    parser.add_argument('--verbose', default=1, type=int, choices=[0, 1])
    parser.add_argument('--max_node_num', default=100000, type=int)
    parser.add_argument('--max_random_steps', default=5, type=int)
    parser.add_argument('--rand_init', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    root_dir = utils.DATA_DIR / 'value_iteration' / 'v3'
    weight_dir = root_dir / 'weights'
    if not weight_dir.is_dir():
        weight_dir.mkdir(parents=True, exist_ok=True)
        if args.rand_init:
            init_model = get_model(args.model)
        else:
            trained_weight_path = utils.DATA_DIR / \
                                  f'weights/explore/ensemble/v1/{args.model.lower()}.pth'
            init_model = get_model(args.model, trained_weight_path)
        init_weight_path = weight_dir / f'{args.model.lower()}_{0:02d}.pth'
        torch.save(init_model.state_dict(), init_weight_path)

    if torch.cuda.is_available():
        gpus = 1
        accelerator = 'dp'
    else:
        gpus = 0
        accelerator = 'cpu'
    default_root_dir = root_dir
    data_dir = root_dir / 'history'

    while True:
        weight_paths = sorted(weight_dir.glob(f'{args.model.lower()}_*.pth'),
                              key=str, reverse=True)
        while len(weight_paths) > args.num_kept_models:
            weight_paths.pop().unlink()
        model_index = int(weight_paths[0].stem[-2:])

        num_batches = len(list(data_dir.glob('*.json')))
        for index in range(num_batches, (model_index + 1) * args.container_size):
            gc.collect()
            prefix = f'{model_index}-{index}'
            SelfPlayPipeline(root_dir, **args.__dict__).run(prefix)
            gc.collect()

        num_samples = len(PolicyPipeline.make_dataset(args.container_size))
        if num_samples >= 10000:
            epochs = 200
        elif num_samples >= 1000:
            epochs = 100
        elif num_samples >= 100:
            epochs = 50
        elif num_samples >= 10:
            epochs = 25
        else:
            epochs = 10
        ppl = PolicyPipeline(**args.__dict__, version='2.0',
                             model_index=model_index, epochs=epochs)
        trainer = pl.Trainer(gpus=gpus, accelerator=accelerator, max_epochs=epochs,
                             default_root_dir=default_root_dir)
        trainer.fit(ppl)


if __name__ == '__main__':
    # main()
    pass