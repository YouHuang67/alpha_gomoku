from alpha_gomoku.cppboard import Board
from alpha_gomoku.gcn import models


def main():
    dim = 128
    embedding = models.PlayerEmbedding(dim)
    gcn = models.GCNII(dim, 128, 6)
    print(gcn(embedding([Board(), Board()])).shape)


if __name__ == '__main__':
    main()