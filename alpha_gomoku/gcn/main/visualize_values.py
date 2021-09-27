import argparse

import torch

from alpha_gomoku import utils
from alpha_gomoku.gcn.train import GCNPipeline
from alpha_gomoku.gcn.train import ValueVisualization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.dir is not None and args.root is not None
    pipeline = ValueVisualization(**args.__dict__)
    pipeline.visualize_values()


if __name__ == '__main__':
    main()
