# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def sequence_mask(lengths, maxlen):
    """
    Args:
        lengths: 1D Tensor, [B]
        maxlen: int

    Returns: 2D Tensor, [B, maxlen], (1 for padding, 0 for true)

    """
    row_vector = torch.arange(0, maxlen, dtype=torch.long)
    matrix = lengths.unsqueeze(-1)
    mask = row_vector >= matrix   # torch.uint8

    return mask


def prepare_input(x, x_lens, y):
    """
    Args:
        x: 2D Tensor, [B, T]
        x_lens: 1D Tensor, [B]
        y: 1D Tensor, [B]

    Returns:
        x_mask

    """
    lens, indices = x_lens.sort(0, descending=True)
    x = x[indices]
    y = y[indices]
    x_lens = x_lens[indices]
    maxlen = x_lens.max()
    x_mask = sequence_mask(x_lens, maxlen)

    return x, x_lens, x_mask, y


def prepare_input2(x, x_lens, y, x_tag):
    """
    Args:
        x: 2D Tensor, [B, T]
        x_lens: 1D Tensor, [B]
        y: 1D Tensor, [B]
        x_tag: 2D Tensor, [B, T]

    Returns:
        x_mask

    """
    maxlen = x_lens.max()

    x = x[:, :maxlen]
    x_tag = x_tag[:, :maxlen]

    lens, indices = x_lens.sort(0, descending=True)
    x = x[indices]
    y = y[indices]
    x_lens = x_lens[indices]
    x_tag = x_tag[indices]

    x_mask = sequence_mask(x_lens, maxlen)

    return x, x_lens, x_mask, y, x_tag


def lr_decay(optimizer, epoch, decay_rate=0.03, init_lr=3e-4):

    lr = init_lr / (1 + decay_rate * epoch)
    print('learning rate: {0}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
