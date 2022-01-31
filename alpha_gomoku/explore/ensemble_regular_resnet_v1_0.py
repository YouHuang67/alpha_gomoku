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


class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        dataset = self.dataset
        if torch.rand(1).item() < 0.5:
            dataset.get_value = False
            board, target, player = dataset[item]
            weight = 0.0
        else:
            dataset.get_value = True
            board, target, player, weight = dataset[item]
        return board, target, player, weight

    def __len__(self):
        return len(self.dataset)


class BoardWrapper(nn.Module):

    def __init__(self, module):
        super(BoardWrapper, self).__init__()
        self.module = module
        self.board_to_tensor = models.BoardToTensor()

    def forward(self, x, *args, **kwargs):
        return self.module(self.board_to_tensor(x), *args, **kwargs)


class ResNetTrainer(pl.LightningModule):

    def __init__(self, **kwargs):
        super(ResNetTrainer, self).__init__()
        self.save_hyperparameters()
        args = self.hparams

        model_cls = {k.lower(): v for k, v in models.__dict__.items()}[args.model.lower()]
        model = model_cls(**utils.get_func_kwargs(model_cls.__init__, kwargs))
        self.model = models.ValueWrapper(BoardWrapper(model))
        self.train_dataset, self.test_dataset = base.VCTActions().split(args.split_ratio)

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr_max,
            momentum=0.9, weight_decay=5e-4
        )
        return [optimizer], [schedule.get_scheduler(args, optimizer)]

    def training_step(self, batch, batch_idx):
        args = self.hparams
        model = self.model
        boards, targets, players, weights = batch

        prob_out, value_out = model((boards, players))

        action_mask = (weights == 0.0)
        if args.label_smoothing:
            labels = F.one_hot(
                targets[action_mask], prob_out.size(-1)
            ).float().to(targets.device)
            labels[labels > 0] = 1 - args.label_smoothing
            labels[labels == 0] = args.label_smoothing / (prob_out.size(-1) - 1)
            action_loss = -(labels * F.log_softmax(
                prob_out[action_mask], dim=-1
            )).sum(-1).mean()
        else:
            action_loss = F.cross_entropy(prob_out[action_mask], targets[action_mask])
        action_acc = (prob_out[action_mask].argmax(-1) == targets[action_mask]).float().mean()
        self.log('action_loss', action_loss, on_step=False, on_epoch=True)
        self.log('action_acc', action_acc, on_step=False, on_epoch=True)

        value_mask = (weights > 0.0)
        value_out = value_out[value_mask]
        targets = targets[value_mask]
        weights = weights[value_mask] ** args.gamma
        weights[targets > 0] *= (targets == 0).float().sum() / (targets > 0).float().sum()
        value_loss = F.binary_cross_entropy_with_logits(value_out, targets.float(), weights)
        roc_auc = roc_auc_score(targets.cpu().numpy(), value_out.detach().cpu().numpy())
        self.log('value_loss', value_loss, on_step=False, on_epoch=True)
        self.log('roc_auc', roc_auc, on_step=False, on_epoch=True)

        return action_loss + value_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        args = self.hparams
        model = self.model
        boards, actions, players = batch
        sample_size = boards.size(0)

        prob_out, pos_value_out = model((boards, players))
        acc = (prob_out.argmax(-1) == actions).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        top_acc = (
            prob_out.argsort(dim=-1, descending=True)[:, :args.top_k] == actions.view(-1, 1)
        ).float().sum(-1).mean()
        self.log(f'top_{args.top_k}_acc', top_acc, on_step=False, on_epoch=True)

        boards.reshape(sample_size, -1).scatter_(
            -1, actions.long().view(-1, 1), players.view(-1, 1)
        )
        _, neg_value_out = model((boards, 1 - players))
        self.log('val_gap', (pos_value_out - neg_value_out).mean(), on_step=False, on_epoch=True)

        roc_auc = roc_auc_score(
            torch.cat([torch.ones(sample_size), torch.zeros(sample_size)]).numpy(),
            torch.cat([pos_value_out.detach().cpu(), neg_value_out.detach().cpu()]).numpy()
        )
        self.log('val_roc_auc', roc_auc, on_step=False, on_epoch=True)

    def train_dataloader(self):
        args = self.hparams
        dataset = self.train_dataset
        dataset.augmentation = True
        dataset = DatasetWrapper(dataset)
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
        weight_dir = utils.DATA_DIR / 'weights' / 'explore' / 'ensemble' / 'regular_resnet' / 'v1'
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
    parser.add_argument('--only_positive_samples', action='store_true')
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
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
    default_root_dir = utils.LOG_DIR / 'explore' / 'ensemble' / 'regular_resnet' / 'v1'
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