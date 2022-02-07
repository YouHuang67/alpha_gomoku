import yaml
import random
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.explore import base
from alpha_gomoku.explore import models
from alpha_gomoku.explore import schedule


class Dataset(base.VCTActions):

    def __init__(self, augmentation=False, dataset=None, only_vct_actions=True):
        super(Dataset, self).__init__(augmentation, dataset)
        self.only_vct_actions = only_vct_actions

    def __getitem__(self, item):
        board_actions, vct_action = self.dataset[item]
        if self.augmentation:
            index = random.randint(0, len(board_actions) - 1)
        else:
            index = 0
        board_actions = board_actions[index]
        vct_action = vct_action[index]

        if self.only_vct_actions:
            action = vct_action
            value = 1.0
        else:
            num_actions = len(board_actions)
            target_index = random.randint(1, num_actions)
            action = (board_actions + (vct_action, ))[target_index]
            board_actions = board_actions[:target_index]
            value = (0.5 - ((num_actions - len(board_actions)) % 2)) * 2

        indices = torch.LongTensor(list(zip(*board_actions)))
        values = torch.LongTensor([(i % 2) - 2 for i in range(len(board_actions))])
        size = torch.Size([self.size, self.size])
        board = torch.sparse.LongTensor(indices, values, size).to_dense() + 2
        player = len(board_actions) % 2

        return board, player, self.action_flatten(*action), value


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


class EnsemblePipeline(pl.LightningModule):

    def __init__(self, **kwargs):
        super(EnsemblePipeline, self).__init__()
        self.save_hyperparameters()
        args = self.hparams

        model_cls = {k.lower(): v for k, v in models.__dict__.items()}[args.model.lower()]
        model = model_cls(**utils.get_func_kwargs(model_cls.__init__, kwargs))
        self.model = nn.Sequential(models.BoardToTensor(), model, EnsembleOutput())
        self.train_dataset, self.test_dataset = Dataset().split(args.split_ratio)

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr_max,
            momentum=0.9, weight_decay=5e-4
        )
        return [optimizer], [schedule.get_scheduler(args, optimizer)]

    def training_step(self, batch, batch_idx):
        model = self.model
        boards, players, actions, values = batch

        pred_log, pred_val = model((boards, players))

        policy_loss = F.cross_entropy(pred_log, actions)
        acc = (pred_log.argmax(-1) == actions).float().mean()
        self.log('policy_loss', policy_loss, on_step=False, on_epoch=True)
        self.log('acc', acc, on_step=False, on_epoch=True)

        value_loss = (pred_val - values).square().mean()
        self.log('value_loss', value_loss, on_step=False, on_epoch=True)

        return policy_loss + value_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        args = self.hparams
        model = self.model
        boards, players, actions, values = batch
        sample_size = boards.size(0)

        pred_log, pred_pos_val = model((boards, players))
        acc = (pred_log.argmax(-1) == actions).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        top_acc = (pred_log.argsort(dim=-1, descending=True)
                   [:, :args.top_k] == actions.view(-1, 1)).float().sum(-1).mean()
        self.log(f'top_{args.top_k}_acc', top_acc, on_step=False, on_epoch=True)

        boards.reshape(sample_size, -1).scatter_(-1, actions.long().view(-1, 1),
                                                 players.view(-1, 1))
        _, pred_neg_val = model((boards, 1 - players))
        self.log('val_gap', (pred_pos_val - pred_neg_val).mean(),
                 on_step=False, on_epoch=True)

        roc_auc = roc_auc_score(torch.cat([torch.ones(sample_size), torch.zeros(sample_size)]).numpy(),
                                torch.cat([pred_pos_val.detach().cpu(),
                                           pred_neg_val.detach().cpu()]).numpy())
        self.log('val_roc_auc', roc_auc, on_step=False, on_epoch=True)

    def train_dataloader(self):
        args = self.hparams
        dataset = self.train_dataset
        dataset.augmentation = True
        dataset.only_vct_actions = False
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=2,
                                           drop_last=True)

    def val_dataloader(self):
        args = self.hparams
        dataset = self.test_dataset
        dataset.augmentation = False
        dataset.only_vct_actions = True
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=2)

    def on_fit_end(self):
        args = self.hparams
        weight_dir = utils.DATA_DIR / 'weights' / 'explore' / 'ensemble' / 'v1'
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
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--top_k', default=5, type=int)
    # advanced setting
    parser.add_argument('--se_reduction', default=1, type=int)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--split_ratio', default=0.25, type=float)
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
    default_root_dir = utils.LOG_DIR / 'explore' / 'ensemble' / 'v1'
    if args.resume:
        assert args.resume_dir is not None
        with open(Path(args.resume_dir) / 'hparams.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            defaults = {k: v for k, v in args.__dict__.items() if k not in config}
        checkpoint_dir = Path(args.resume_dir) / 'checkpoints'
        checkpoint_path = sorted(checkpoint_dir.glob('*'))[-1]
        ppl = EnsemblePipeline.load_from_checkpoint(checkpoint_path, **defaults)
        max_epochs = ppl.hparams.epochs
        trainer = pl.Trainer(gpus=gpus, accelerator=accelerator,
                             max_epochs=max_epochs, default_root_dir=default_root_dir,
                             resume_from_checkpoint=checkpoint_path)
    else:
        ppl = EnsemblePipeline(**args.__dict__, version='1.0',
                               description='only train one resnet')
        max_epochs = ppl.hparams.epochs
        trainer = pl.Trainer(gpus=gpus, accelerator=accelerator,
                             max_epochs=max_epochs,
                             default_root_dir=default_root_dir)
    trainer.fit(ppl)


if __name__ == '__main__':
    main()