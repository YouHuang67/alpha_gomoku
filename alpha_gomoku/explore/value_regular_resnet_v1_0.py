import yaml
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from alpha_gomoku import utils
from alpha_gomoku.explore import base
from alpha_gomoku.explore import models
from alpha_gomoku.explore import schedule


class ResNetTrainer(pl.LightningModule):

    def __init__(self, **kwargs):
        super(ResNetTrainer, self).__init__()
        self.save_hyperparameters()
        args = self.hparams

        model_cls = {k.lower(): v for k, v in models.__dict__.items()}[args.model.lower()]
        model = model_cls(**utils.get_func_kwargs(model_cls.__init__, kwargs))
        self.model = nn.Sequential(models.BoardToTensor(), model, nn.Linear(15**2, 1, bias=True))
        self.train_dataset, self.test_dataset = base.VCTActions().split(args.split_ratio)

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr_max,
            momentum=0.9, weight_decay=5e-4
        )
        return [optimizer], [schedule.get_scheduler(args, optimizer)]

    def training_step(self, batch, batch_idx):
        model = self.model
        boards, targets, players, weights = batch
        positive_mask = (targets > 0)
        white_mask = players > 0
        positive_min = min(len(targets[positive_mask & (~white_mask)]),
                           len(targets[positive_mask & white_mask]))
        negative_min = min(len(targets[(~positive_mask) & (~white_mask)]),
                           len(targets[(~positive_mask) & white_mask]))
        boards = torch.cat([
            boards[positive_mask & (~white_mask)][:positive_min],
            boards[positive_mask & white_mask][:positive_min],
            boards[(~positive_mask) & (~white_mask)][:negative_min],
            boards[(~positive_mask) & white_mask][:negative_min]
        ], dim=0)
        targets = torch.cat([
            targets[positive_mask & (~white_mask)][:positive_min],
            targets[positive_mask & white_mask][:positive_min],
            targets[(~positive_mask) & (~white_mask)][:negative_min],
            targets[(~positive_mask) & white_mask][:negative_min]
        ], dim=0)
        players = torch.cat([
            players[positive_mask & (~white_mask)][:positive_min],
            players[positive_mask & white_mask][:positive_min],
            players[(~positive_mask) & (~white_mask)][:negative_min],
            players[(~positive_mask) & white_mask][:negative_min]
        ], dim=0)
        weights = torch.cat([
            weights[positive_mask & (~white_mask)][:positive_min],
            weights[positive_mask & white_mask][:positive_min],
            weights[(~positive_mask) & (~white_mask)][:negative_min],
            weights[(~positive_mask) & white_mask][:negative_min]
        ], dim=0)
        criterion = nn.BCEWithLogitsLoss(weights ** self.hparams.gamma)

        out = model((boards, players)).view(-1)
        loss = criterion(out, targets.float())
        self.log('loss', loss, on_step=False, on_epoch=True)

        roc_auc = roc_auc_score(targets.cpu().numpy(), out.detach().cpu().numpy())
        self.log('roc_auc', roc_auc, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        model = self.model
        boards, actions, players = batch
        sample_size = boards.size(0)

        pos_out = model((boards, players)).view(-1)

        boards.reshape(sample_size, -1).scatter_(
            -1, actions.long().view(-1, 1), players.view(-1, 1)
        )
        neg_out = model((boards, 1 - players)).view(-1)

        self.log('val_gap', (pos_out - neg_out).mean(), on_step=False, on_epoch=True)

        roc_auc = roc_auc_score(
            torch.cat([torch.ones(sample_size), torch.zeros(sample_size)]).numpy(),
            torch.cat([pos_out.detach().cpu(), neg_out.detach().cpu()]).numpy()
        )
        self.log('val_roc_auc', roc_auc, on_step=False, on_epoch=True)

    def train_dataloader(self):
        args = self.hparams
        dataset = self.train_dataset
        dataset.augmentation = True
        dataset.get_value = True
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        args = self.hparams
        dataset = self.test_dataset
        dataset.augmentation = False
        dataset.get_value = False
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

    def on_fit_end(self):
        args = self.hparams
        weight_dir = utils.DATA_DIR / 'weights' / 'explore' / 'value' / 'regular_resnet' / 'v1'
        weight_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weight_dir / f'{args.model.lower()}_kl_{args.kernel_one_level}.pth'
        model = nn.Sequential(self.model[1], self.model[2])
        torch.save(model.state_dict(), weight_path)


def parse_args():
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='SEWideResNet16_1', type=str)
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
    # train
    parser.add_argument('--gamma', default=0.5, type=float)
    # advanced setting
    parser.add_argument('--kernel_one_level', default=0, type=int)
    parser.add_argument('--se_reduction', default=1, type=int)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--split_ratio', default=0.75, type=float)
    parser.add_argument('--only_positive_samples', action='store_true')
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
    default_root_dir = utils.LOG_DIR / 'explore' / 'value' / 'regular_resnet' / 'v1'
    if args.resume:
        assert args.resume_dir is not None
        with open(Path(args.resume_dir) / 'hparams.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            defaults = {k: v for k, v in args.__dict__.items() if k not in config}
        checkpoint_dir = Path(args.resume_dir) / 'checkpoints'
        checkpoint_path = sorted(checkpoint_dir.glob('*'))[-1]
        ppl = ResNetTrainer.load_from_checkpoint(checkpoint_path, **defaults)
        max_epochs = ppl.hparams.epochs
        trainer = pl.Trainer(
            gpus=gpus, accelerator=accelerator, max_epochs=max_epochs,
            default_root_dir=default_root_dir, resume_from_checkpoint=checkpoint_path
        )
    else:
        ppl = ResNetTrainer(**args.__dict__, version='1.0',
                            description='only train one resnet')
        max_epochs = ppl.hparams.epochs
        trainer = pl.Trainer(
            gpus=gpus, accelerator=accelerator, max_epochs=max_epochs,
            default_root_dir=default_root_dir
        )
    trainer.fit(ppl)


if __name__ == '__main__':
    main()