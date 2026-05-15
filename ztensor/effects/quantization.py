import torch


def downsample(plane: torch.Tensor):
    plane = plane / 2

    return plane


def upsample(plane: torch.Tensor):
    plane = plane * 2

    return plane