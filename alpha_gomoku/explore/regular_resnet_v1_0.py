import yaml
import argparse
from pathlib import Path

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
        self.model = nn.Sequential(models.BoardToTensor(), model)
        self.train_dataset, self.test_dataset = base.VCTActions().split(args.split_ratio)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def configure_optimizers(self):
        args = self.hparams
        optimizer_cls = {
            k.lower(): v for k, v in torch.optim.__dict__.items()
            if isinstance(v, type) and issubclass(v, torch.optim.Optimizer)
        }[args.optimizer.lower()]
        kwargs = {'lr': args.lr_max, 'weight_decay': 5e-4}
        if optimizer_cls == torch.optim.SGD:
            kwargs['momentum'] = 0.9
        optimizer = optimizer_cls(self.model.parameters(), **kwargs)
        if args.lr_schedule.lower() == 'none':
            scheduler = []
        else:
            scheduler = [schedule.get_scheduler(args, optimizer)]
        return [optimizer], scheduler

    def training_step(self, batch, batch_idx):
        args = self.hparams
        model = self.model
        boards, actions, players = batch
        sample_size = boards.size(0)

        if args.only_positive_samples:
            mask = torch.zeros(sample_size).to(boards.device) != 0
        else:
            mask = torch.rand(sample_size).to(boards.device) < 0.5
        if mask.any():
            negative_boards = boards[mask].reshape(mask.long().sum().item(), -1)
            indices = actions[mask].long().view(-1, 1)
            negative_boards.scatter_(-1, indices, players[mask].view(-1, 1))
            boards[mask] = negative_boards.reshape(-1, *boards.shape[1:])
            players[mask] = 1 - players[mask]

        out = model((boards, players))
        acc = (out[~mask].argmax(-1) == actions[~mask]).float().mean()
        self.log('acc', acc, on_step=False, on_epoch=True)

        actions[mask] = out.argmax(-1)[mask]
        signs = torch.ones_like(mask).float()
        signs[mask] = -signs[mask]
        loss = (signs * self.criterion(out, actions)).mean()
        self.log('loss', loss, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        args = self.hparams
        model = self.model
        boards, actions, players = batch
        sample_size = boards.size(0)

        out = model((boards, players))
        acc = (out.argmax(-1) == actions).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        top_acc = (
            out.argsort(dim=-1, descending=True)[:, :args.top_k] == actions.view(-1, 1)
        ).float().sum(-1).mean()
        self.log(f'top_{args.top_k}_acc', top_acc, on_step=False, on_epoch=True)

        pos_prob = torch.gather(F.softmax(out, dim=-1), -1, actions.view(-1, 1)).mean()
        self.log('pos_prob', pos_prob, on_step=False, on_epoch=True)

        indices = actions.long().view(-1, 1)
        boards.reshape(sample_size, -1).scatter_(-1, indices, players.view(-1, 1))
        neg_prob = F.softmax(model((boards, 1 - players)), dim=-1).max(-1)[0].mean()
        self.log('neg_prob', neg_prob, on_step=False, on_epoch=True)

    def train_dataloader(self):
        args = self.hparams
        dataset = self.train_dataset
        dataset.augmentation = True
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        args = self.hparams
        dataset = self.test_dataset
        dataset.augmentation = False
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

    def on_fit_end(self):
        args = self.hparams
        weight_dir = utils.DATA_DIR / 'weights' / 'explore' / 'regular_resnet' / 'v1'
        weight_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weight_dir / f'{args.model.lower()}_kl_{args.kernel_one_level}.pth'
        torch.save(self.model[1].state_dict(), weight_path)


def parse_args():
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='PreActResNet50', type=str)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr_schedule', default='piecewise',
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
    parser.add_argument('--kernel_one_level', default=0, type=int)
    parser.add_argument('--se_reduction', default=1, type=int)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--split_ratio', default=0.25, type=float)
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
    default_root_dir = utils.LOG_DIR / 'explore' / 'regular_resnet' / 'v1'
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