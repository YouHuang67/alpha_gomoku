import time
import argparse

import torch

from alpha_gomoku import utils
from alpha_gomoku.explore import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    # model parameters
    parser.add_argument('--kernel_one_level', default=0, type=int)
    # setting
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    model_cls = {
        k.lower(): v for k, v in models.__dict__.items()
    }[args.model.lower()]
    model = model_cls(
        **utils.get_func_kwargs(model_cls.__init__, args.__dict__)
    ).eval()
    x = torch.rand(args.batch_size, 3, 15, 15)
    with torch.no_grad():
        model(x)
        start = time.time()
        for _ in range(args.steps):
            model(x)
        end = time.time()
        total_time = end - start
    print(f'model: {args.model}: ')
    print(f'average time of each step with {x.size(0)} samples: {total_time/args.steps:.4f}s')
    print(f'average time of each sample: {total_time/args.steps/x.size(0):.4f}s')


if __name__ == '__main__':
    main()




