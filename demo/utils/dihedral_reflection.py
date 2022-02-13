import torch
from torchvision.transforms import functional as TF


def rotate0(x):
    return x


def inv_rotate0(x):
    return x


def rotate90(x):
    return TF.rotate(x, 90)


def inv_rotate90(x):
    return TF.rotate(x, 270)


def rotate180(x):
    return TF.rotate(x, 180)


def inv_rotate180(x):
    return TF.rotate(x, 180)


def rotate270(x):
    return TF.rotate(x, 270)


def inv_rotate270(x):
    return TF.rotate(x, 90)


def vertical_flip(x):
    return torch.flip(x, [-2])


def inv_vertical_flip(x):
    return torch.flip(x, [-2])


def horizontal_flip(x):
    return torch.flip(x, [-1])


def inv_horizontal_flip(x):
    return torch.flip(x, [-1])


def diagonal_flip(x):
    return x.transpose(-1, -2)


def inv_diagonal_flip(x):
    return x.transpose(-1, -2)


def off_diagonal_flip(x):
    return inv_rotate90(rotate90(x).transpose(-1, -2))


def inv_off_diagonal_flip(x):
    return inv_rotate90(rotate90(x).transpose(-1, -2))


REFLECTION_FUNCS = [rotate0, rotate90, rotate180, rotate270, vertical_flip,
                    horizontal_flip, diagonal_flip, off_diagonal_flip]
INVERSE_FUNCS = [globals()[f'inv_{func.__name__}'] for func in REFLECTION_FUNCS]