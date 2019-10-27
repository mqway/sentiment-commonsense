# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from core.nn.encoder import TokenEmbedding, BiLstmEncoder, MaxPooling


__all__ = ['BiLSTMMax']


class BiLSTMMax(nn.Module):
    def __init__(self, vocab_size, word_dim, lstm_dim, mlp_dim, n_class, we=None):
        super(BiLSTMMax, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.mlp_dim = mlp_dim
        self.n_class = n_class

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)
        self.bilstm = BiLstmEncoder(self.word_dim, self.lstm_dim)
        self.max_pooling_layer = MaxPooling()
        self.mlp_layer = nn.Linear(2 * self.lstm_dim, self.mlp_dim)
        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

    def forward(self, word_inputs, mask, seq_lengths):

        x_embedding = self.word_embedding(word_inputs)

        hs = self.bilstm(x_embedding, seq_lengths)

        sentence = self.max_pooling_layer(hs, mask)

        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)

        return logits
