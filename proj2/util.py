import torch
import math


def generate_samples(count, one_hot=False):
    input = torch.empty((count, 2)).uniform_(0, 1)
    if one_hot:
        target = torch.empty((count, 2)).zero_()
        mask = input.pow(2).sum(1) < 2 / math.pi
        target[~mask, 0] = 1
        target[mask, 1] = 1
    else:
        target = input.pow(2).sum(1) < 2 / math.pi
    return input, target.long()
