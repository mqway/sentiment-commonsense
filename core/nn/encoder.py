# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['TokenEmbedding', 'BiLstmEncoder', 'SelfAttention', 'MaxPooling', 'AveragePooling']


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, freeze=False, pretrained_emb=None):
        super(TokenEmbedding, self).__init__()
        if pretrained_emb is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_emb), freeze=freeze)
        else:
            self.embedding = self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, x):
        """
        Args:
            x: 2D Tensor, [B, T]

        Returns: 3D Tensor,  [B, T, D]

        """
        x = self.embedding(x)
        return x


class BiLstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(BiLstmEncoder, self).__init__()

        self.rnn_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, x, x_len):
        """
        Args:
            x: 3D Tensor, [B, T, D]
            x_len: 1D Tensor, [B]

        Returns: [B, T, 2*D]

        """

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        packed_input = pack_padded_sequence(x, x_len, batch_first=True)

        # packed_output: [B, T, H], H = rnn_size * num_directions
        # Final time-step hidden state (h_n) of the LSTM
        # _h_n: [num_layers * num_directions, B, rnn_size]
        # _c_n: [num_layers * num_directions, B, rnn_size]
        packed_output, (_h_n, _c_n) = self.rnn(packed_input)

        # hs: [B, T, H]
        hs, _ = pad_packed_sequence(packed_output, batch_first=True)
        # https://blog.csdn.net/appleml/article/details/80143212
        hs = hs.contiguous()

        return hs


class SelfAttention(nn.Module):
    """ Standard Self attention over a sequence:

    """

    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.linear_x = nn.Linear(input_size, hidden_size)
        self.linear_u = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        stdv = 6. / math.sqrt(sum(self.linear_x.weight.data.shape))
        self.linear_x.weight.data.uniform_(-stdv, stdv)
        self.linear_x.bias.data.fill_(0)

        stdv = 6. / math.sqrt(sum(self.linear_u.weight.data.shape))
        self.linear_u.weight.data.uniform_(-stdv, stdv)
        self.linear_u.bias.data.fill_(0)

    def forward(self, x, x_mask):
        """
        Args:
            x: 3D Tensor, [B, T, D]
            x_mask: 2D Tensor, [B, T], (1 for padding, 0 for true)

        Returns:
            alpha: 2D Tensor, [B, T]
            y: 2D Tensor, [B, D]
        """
        b, t, d = x.shape

        v = self.linear_x(x)  # [B, T, H]
        v = torch.tanh(v)   # [B, T, H]

        scores = self.linear_u(v).view(b, t)

        masked_scores = scores.masked_fill(x_mask, -float('inf'))

        alpha = F.softmax(masked_scores, dim=-1)

        x = x.transpose(1, 2)
        y = torch.bmm(x, alpha.unsqueeze(2)).squeeze(2)

        return y, alpha


class MaxPooling(nn.Module):

    def forward(self, x, x_mask):
        """
        Args:
            x: 3D Tensor, [B, T, D]
            x_mask: 2D Tensor, [B, T], (1 for padding, 0 for true)

        Returns: 2D Tensor, [B, D]

        """
        mask = x_mask.float().masked_fill(x_mask, -float('inf'))  # [B, T]
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        x = x + mask  # [B, T, D]
        x, _ = x.max(dim=1)  # [B, D]
        return x


class AveragePooling(nn.Module):

    def forward(self, x, x_mask):
        """
        Args:
            x: 3D tensor, [B, T, D]
            x_mask: 2D tensor, [B, T], (1 for padding, 0 for true)

        Returns: 2D Tensor, [B, D]

        """
        batch_len = x_mask.size(1)
        x_len = batch_len - x_mask.float().sum(dim=1, keepdim=True)  # [B, 1]

        x = x.sum(dim=1) / x_len  # [B, D]
        return x
