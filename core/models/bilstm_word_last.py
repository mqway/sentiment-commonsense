# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from core.nn.encoder import TokenEmbedding
from torch.nn.utils.rnn import pack_padded_sequence

__all__ = ['BiLSTMLast']


class BiLSTMLast(nn.Module):
    def __init__(self, vocab_size, word_dim, lstm_dim, mlp_dim, n_class, we=None):
        super(BiLSTMLast, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.mlp_dim = mlp_dim
        self.n_class = n_class

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)
        self.bilstm = nn.LSTM(self.word_dim, self.lstm_dim, batch_first=True, bidirectional=True)
        self.mlp_layer = nn.Linear(2 * self.lstm_dim, self.mlp_dim)
        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

    def forward(self, word_inputs, mask, seq_lengths):

        x_embedding = self.word_embedding(word_inputs)  # [B, T, D]

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        packed_input = pack_padded_sequence(x_embedding, seq_lengths, batch_first=True)

        # packed_output: [B, T, H], H = rnn_size * num_directions
        # Final time-step hidden state (hn) of the LSTM
        # hn: [num_layers * num_directions, B, rnn_size]
        # cn: [num_layers * num_directions, B, rnn_size]

        packed_output, (hn, cn) = self.bilstm(packed_input)

        # use Last time step of BiLSTM as final sentence representation
        batch_size = word_inputs.size(0)
        sentence = hn.transpose(0, 1).contiguous().view(batch_size, 2*self.lstm_dim)

        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)

        return logits
