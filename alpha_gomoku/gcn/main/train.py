import argparse

import torch

from alpha_gomoku import utils
from alpha_gomoku.gcn.train import GCNPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1024)
    # dataset
    parser.add_argument('--root', type=str, default='F:/repositories/gomocup/records')
    parser.add_argument('--split', type=float, default=0.75)
    parser.add_argument('--train_dir', type=str, default='_temp_tensors/train')
    parser.add_argument('--test_dir', type=str, default='_temp_tensors/test')
    parser.add_argument('--num_workers', type=int, default=4)
    # model
    parser.add_argument('--embedding', type=str, default='PlayerEmbedding')
    parser.add_argument('--network', type=str, default='GraphConvolutionNetwork')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=64)
    # Residual
    parser.add_argument('--block_num', type=int, default=8)
    # GCNII
    parser.add_argument('--layer_num', type=int, default=10)
    parser.add_argument('--residual', type=utils.str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    # train
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--value_weight', type=float, default=1.0)
    return parser.parse_args()


def main():
    # torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    GCNPipeline(**args.__dict__).train()


if __name__ == '__main__':
    main()
