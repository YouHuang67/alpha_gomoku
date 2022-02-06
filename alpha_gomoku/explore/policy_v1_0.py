import yaml
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.explore import base
from alpha_gomoku.explore import models
from alpha_gomoku.explore import schedule
from alpha_gomoku.datasets import VCTDefenseActions


class Dataset(torch.utils.data.Dataset):
    size = Board.BOARD_SIZE

    def __init__(self, dataset, augmentation):
        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        previous_actions, actions = self.dataset[item]

        previous_actions = previous_actions[0]
        indices = torch.LongTensor(list(zip(*previous_actions)))
        values = torch.LongTensor([(i % 2) - 2 for i in range(len(previous_actions))])
        size = torch.Size([self.size, self.size])
        board = torch.sparse.LongTensor(indices, values, size).to_dense() + 2
        player = len(previous_actions) % 2

        indices = torch.LongTensor(list(zip(*actions)))
        target = torch.sparse.LongTensor(
            indices, torch.ones(len(actions)).long(), size
        ).to_dense()

        if self.augmentation:
            aug_func = random.choice(list(utils.AUGMENTATION_FUNCS.values()))
            board = aug_func(board.unsqueeze(0)).squeeze(0)
            target = aug_func(target.unsqueeze(0)).squeeze(0)
        return board, player, target.reshape(-1)


class PolicyPipeline(pl.LightningModule):

    def __init__(self, **kwargs):
        super(PolicyPipeline, self).__init__()
        self.save_hyperparameters()
        args = self.hparams

        model_cls = {k.lower(): v for k, v in models.__dict__.items()}[args.model.lower()]
        model = model_cls(**utils.get_func_kwargs(model_cls.__init__, kwargs))
        self.model = nn.Sequential(models.BoardToTensor(), model)
        self.train_dataset, self.test_dataset = VCTDefenseActions(
            utils.DATA_DIR / 'gomocup' / 'records'
        ).split(args.split_ratio)

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr_max,
            momentum=0.9, weight_decay=5e-4
        )
        return [optimizer], [schedule.get_scheduler(args, optimizer)]

    @staticmethod
    def top_k(out, target, k):
        row = torch.arange(out.size(0)).view(-1, 1).repeat(1, k).to(out.device)
        col = out.argsort(-1, descending=True)[:, :k]
        indices = torch.stack([row.reshape(-1), col.reshape(-1)], 0)
        result = torch.sparse.LongTensor(
            indices, torch.ones(k * out.size(0)).long().to(out.device), out.size()
        ).to_dense().to(out.device)
        return ((target * result).sum(-1) > 0).float().mean()

    def training_step(self, batch, batch_idx):
        args = self.hparams
        model = self.model
        boards, players, targets = batch

        out = model((boards, players))
        loss = -(targets * F.softmax(out, -1)).sum(-1).log().mean()
        self.log('loss', loss, on_step=False, on_epoch=True)

        self.log('acc', self.top_k(out, targets, 1),
                 on_step=False, on_epoch=True)
        self.log('top_k', self.top_k(out, targets, args.top_k),
                 on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        args = self.hparams
        model = self.model
        boards, players, targets = batch

        out = model((boards, players))
        self.log('val_acc', self.top_k(out, targets, 1),
                 on_step=False, on_epoch=True)
        self.log('val_top_k', self.top_k(out, targets, args.top_k),
                 on_step=False, on_epoch=True)

    def train_dataloader(self):
        args = self.hparams
        dataset = Dataset(self.train_dataset, True)
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        args = self.hparams
        dataset = Dataset(self.test_dataset, False)
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

    def on_fit_end(self):
        args = self.hparams
        weight_dir = utils.DATA_DIR / 'weights' / 'explore' / 'policy' / 'v1'
        weight_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weight_dir / f'{args.model.lower()}.pth'
        torch.save(self.model.state_dict(), weight_path)


def parse_args():
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str)
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
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--top_k', default=5, type=int)
    # advanced setting
    parser.add_argument('--se_reduction', default=1, type=int)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--split_ratio', default=0.9, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    if torch.cuda.is_available():
        gpus = 1
        accelerator = 'dp'
    else:
        gpus = 0
        accelerator = 'cpu'
    default_root_dir = utils.LOG_DIR / 'explore' / 'policy' / 'v1'
    if args.resume:
        assert args.resume_dir is not None
        with open(Path(args.resume_dir) / 'hparams.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            defaults = {k: v for k, v in args.__dict__.items() if k not in config}
        checkpoint_dir = Path(args.resume_dir) / 'checkpoints'
        checkpoint_path = sorted(checkpoint_dir.glob('*'))[-1]
        ppl = PolicyPipeline.load_from_checkpoint(checkpoint_path, **defaults)
        max_epochs = ppl.hparams.epochs
        trainer = pl.Trainer(
            gpus=gpus, accelerator=accelerator, max_epochs=max_epochs,
            default_root_dir=default_root_dir, resume_from_checkpoint=checkpoint_path
        )
    else:
        ppl = PolicyPipeline(**args.__dict__, version='1.0',
                             description='train policy function')
        max_epochs = ppl.hparams.epochs
        trainer = pl.Trainer(
            gpus=gpus, accelerator=accelerator, max_epochs=max_epochs,
            default_root_dir=default_root_dir
        )
    trainer.fit(ppl)


if __name__ == '__main__':
    main()