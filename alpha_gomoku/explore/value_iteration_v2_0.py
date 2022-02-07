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
from alpha_gomoku.explore import models
from alpha_gomoku.explore import schedule
# from alpha_gomoku.mcts.mcts_v1 import Node
from alpha_gomoku.mcts.mcts_v2 import Node


class NearArea(nn.Module):
    size = Board.BOARD_SIZE

    def __init__(self, gap=3):
        super(NearArea, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(gap * 2 + 1),
                              stride=1, padding=gap, bias=False)
        self.conv.weight.data.copy_(torch.ones_like(self.conv.weight))
        self.conv.weight.requires_grad_(False)

    def forward(self, legal_masks):
        masks = (~legal_masks)
        zero_masks = (masks.long().reshape(-1, self.size ** 2).sum(-1) == 0)
        if zero_masks.any():
            masks[zero_masks, self.size // 2, self.size // 2] = True
        near_areas = self.conv(masks.float().reshape(-1, 1, self.size, self.size))
        return (near_areas > 0.0).view(-1, self.size ** 2).detach().cpu()


class MCTS(object):
    size = Board.BOARD_SIZE
    num_stones = Board.BOARD_SIZE ** 2
    device = utils.DEVICE
    near_area = NearArea(gap=3).to(device)

    def __init__(self, model, boards, visit_times=200, cpuct=1.0,
                 verbose=1, max_node_num=100000, **kwargs):
        self.__dict__.update({k: v for k, v in locals().items()
                              if k not in ['self', 'boards']})
        self.model = model.to(self.device)
        self.roots = []
        self.node_tables = []
        for index, board in enumerate(boards):
            node_table = OrderedDict()
            root = Node.get_node(board, node_table, cpuct=cpuct, index=index)
            self.roots.append(root)
            self.node_tables.append(node_table)

    @classmethod
    def boards_to_tensors(cls, boards):
        board_tensors = [torch.LongTensor(board.vector) for board in boards]
        board_tensors = torch.stack(board_tensors, dim=0)
        board_tensors = board_tensors.reshape(-1, cls.size, cls.size)
        players = torch.LongTensor([board.player for board in boards])
        legal_masks = (board_tensors == 2)
        return board_tensors.to(cls.device), \
               players.to(cls.device), legal_masks.to(cls.device)

    @classmethod
    def flatten(cls, *args):
        row, col = (args[0] if len(args) == 1 else args)
        return row * cls.size + col

    @classmethod
    def unflatten(cls, action):
        return (action // cls.size, action % cls.size)

    @torch.no_grad()
    def predict(self, board_tensors, players, legal_masks):
        funcs = list(utils.AUGMENTATION_FUNCS.values())
        board_tensors = torch.cat([func(board_tensors) for func in funcs], dim=0)
        players = torch.cat([players for func in funcs], dim=0)

        logits, values = self.model((board_tensors, players))

        logits = logits.reshape(len(funcs), -1, self.size, self.size)
        logits = torch.stack([func(log) for func, log in
                              zip(utils.INVERSE_FUNCS.values(), logits)], dim=0)
        logits = logits.mean(0)
        logits[~legal_masks] = -torch.inf
        probs = F.softmax(logits.view(-1, self.num_stones), dim=-1)

        values = values.reshape(len(funcs), -1).mean(0)
        return probs.cpu().detach(), values.cpu().detach()

    def evaluate_actions_and_values(self, boards):
        prob_list = []
        value_list = []
        inputs = self.boards_to_tensors(boards)
        for board, probs, value, near_area in zip(boards, *self.predict(*inputs),
                                                  self.near_area(inputs[-1])):
            actions = board.evaluate(1)
            if len(actions):
                actions = list(map(self.flatten, actions))
            else:
                actions = [action for action, value in enumerate(board.vector)
                           if value == 2 and near_area[action].item()]
            probs = {self.unflatten(action): probs[action].item() for action in actions}
            prob_sum = sum(probs.values())
            prob_list.append({act: prob / prob_sum for act, prob in probs.items()})
            value_list.append(value.item())
        return prob_list, value_list

    def search(self, description=''):
        final_actions = []
        roots = []
        indices = []
        for index, root in enumerate(self.roots):
            board = root.board
            actions = board.copy().evaluate(self.max_node_num)
            if len(actions) and board.attacker == board.player:
                final_actions.append(random.choice(actions))
            else:
                final_actions.append(None)
                roots.append(root)
                indices.append(index)

        if len(roots) == 0:
            return final_actions

        iterator = range(self.visit_times)
        if self.verbose:
            iterator = tqdm(iterator, description)
        for _ in iterator:
            nodes = []
            depths = []
            boards = []
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
                boards.append(board)

            if len(boards) == 0:
                continue

            prob_list, value_list = self.evaluate_actions_and_values(boards)

            for node, probs, board, value, depth in zip(nodes, prob_list, boards,
                                                        value_list, depths):
                node.expand(probs, board, self.node_tables[node.index])
                node.backward(value, depth)

        for index, root in zip(indices, roots):
            children = list(root.children.items())
            if len(children):
                random.shuffle(children)
                final_actions[index] = sorted(
                    children, key=lambda x: x[1].visit
                )[-1][0]
            else:
                final_actions[index] = root.attack_action
        return final_actions

    def move(self, actions):
        roots = self.roots
        self.roots = []
        for root, action in list(zip(roots, actions)):
            index = root.index
            board = root.board.copy().move(action)
            if board.is_over:
                continue
            root = Node.get_node(board, self.node_tables[index],
                                 cpuct=self.cpuct, index=index)
            root.board = board
            self.roots.append(root)


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


def get_model(backbone, weight_path=''):
    backbone_cls = {k.lower(): v for k, v in models.__dict__.items()}[backbone.lower()]
    model = nn.Sequential(models.BoardToTensor(), backbone_cls(), EnsembleOutput())
    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    return model.eval()


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

        self.root_dir = utils.DATA_DIR / 'value_iteration' / 'v2'
        self.data_dir = self.root_dir / 'history'
        self.weight_dir = self.root_dir / 'weights'
        weight_path = self.weight_dir / f'{args.model.lower()}_{args.model_index:02d}.pth'
        self.model = get_model(args.model, weight_path)

        self.dataset = self.make_dataset(args.container_size)

    @staticmethod
    def make_dataset(container_size):
        history_list = []
        root_dir = utils.DATA_DIR / 'value_iteration' / 'v2'
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
    root_dir = utils.DATA_DIR / 'value_iteration' / 'v2'
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
            SelfPlayPipeline(root_dir, args.model,
                             args.num_boards, **args.__dict__).run(prefix)
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
    main()