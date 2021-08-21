import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alpha_gomoku import utils
from alpha_gomoku.cppboard import Board
from alpha_gomoku.datasets import piskvork
from alpha_gomoku.gcn.models import PlayerEmbedding
from alpha_gomoku.gcn.models import GraphConvolutionNetwork


class VCTDataset(piskvork.PiskvorkVCTActions):
    
    def __init__(self, embedding, *args, **kwargs):
        self.embedding = embedding
        super(VCTDataset, self).__init__(*args, **kwargs)
        
    def __getitem__(self, item):
        actions, vct_action = super(VCTDataset, self).__getitem__(item)
        embedding = self.embedding(Board(actions))[0]
        vct_action = vct_action[0] * Board.BOARD_SIZE + vct_action[1]
        return embedding, vct_action


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--piskvork', type=str, default='F:/repositories/gomocup/records')
    # model
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--block_num', type=int, default=4)
    # train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    piskvork_dir = Path(args.piskvork)
    vct_path = piskvork_dir / 'vct_actions.json'
    embedding = PlayerEmbedding(args.dim)
    if vct_path.is_file():
        dataset = VCTDataset(embedding)
        dataset.load(vct_path)
    else:
        dataset = VCTDataset(embedding, piskvork_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    model = GraphConvolutionNetwork(args.dim, args.hidden_dim, args.block_num)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedding.parameters()), 
        # model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    sample_num = 0.0
    model.train()
    for epoch in range(1, args.epochs + 1):
        with tqdm(dataloader, desc=f'epoch {epoch}/{args.epochs}: ') as pbar:
            for boards, actions in pbar:
                pred = model(boards)
                loss = ce_loss(pred, actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sample_num += pred.size(0)
                total_loss += pred.size(0) * loss.item()
                total_acc += (pred.argmax(-1) == actions.long()).sum()
                info = f'epoch {epoch}/{args.epochs}: '
                info += f'loss: {total_loss/sample_num:.4f} '
                info += f'acc: {total_acc/sample_num:.4f}'
                pbar.set_description(info)


if __name__ == '__main__':
    main()
