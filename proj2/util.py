import torch
import math


def generate_samples(count):
    input_ = torch.empty((count, 2)).uniform_(0, 1)
    label = (input_.pow(2).sum(1) < 1 / (2 * math.pi)).type(torch.long)
    return input_, label
