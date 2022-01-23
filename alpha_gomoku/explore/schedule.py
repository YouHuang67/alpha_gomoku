import numpy as np

import torch


def get_scheduler(args, optimizer):
    if args.lr_schedule == 'superconverge':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
        )
    elif args.lr_schedule == 'piecewise':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [int(0.5 * args.epochs), int(0.75 * args.epochs)], gamma=0.1
        )
    elif args.lr_schedule == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda t: np.interp(
                [t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                [args.lr_max, args.lr_max, args.lr_max / 10,  args.lr_max / 100]
            )[0]
        )
    elif args.lr_schedule == 'onedrop':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda t: args.lr_max if t < args.lr_drop_epoch else args.lr_one_drop
        )
    elif args.lr_schedule == 'multipledecay':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda t: args.lr_max - (t // (args.epochs // 10)) * (args.lr_max / 10)
        )
    elif args.lr_schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda t: args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
        )
    return lr_scheduler