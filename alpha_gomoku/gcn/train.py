import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import utils
from gcn.models import GraphConvolutionNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--piskvork', type=str)
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == '__main__':
    main()
