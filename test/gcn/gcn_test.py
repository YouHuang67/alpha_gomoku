from alpha_gomoku.cppboard import Board
from alpha_gomoku.gcn import models


def main():
    dim = 128
    embedding = models.PlayerEmbedding(dim)
    gcn = models.GraphConvolutionNetwork(dim, 128, 4)
    outs = gcn(embedding([Board(), Board()]))
    print(outs[0].shape, outs[1].shape)


if __name__ == '__main__':
    main()