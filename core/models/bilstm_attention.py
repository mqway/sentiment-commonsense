# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from core.nn.encoder import TokenEmbedding, BiLstmEncoder, SelfAttention


class BilstmClassifier(nn.Module):
    def __init__(self, vocab_size, word_dim, lstm_dim, mlp_dim, n_class, we):
        super(BilstmClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.mlp_dim = mlp_dim
        self.n_class = n_class
        self.drop = nn.Dropout(p=0.5)

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)
        self.bilstm = BiLstmEncoder(self.word_dim, self.lstm_dim)

        self.interaction_layer = SelfAttention(2*self.lstm_dim, self.lstm_dim)

        self.mlp_layer = nn.Linear(2 * self.lstm_dim, self.mlp_dim)
        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

    def forward(self, x, x_len, x_mask, return_alpha=False):
        """
        Args:
            x: 2D tensor, [B, T]
            x_len: 1D tensor, [B]
            x_mask: 2D tensor, [B, T]
            return_alpha: boolean

        Returns:

        """
        x_embedding = self.word_embedding(x)
        x_embedding = self.drop(x_embedding)

        hs = self.bilstm(x_embedding, x_len)  # [B, T, 2*lstm_dim]

        sentence, _alpha = self.interaction_layer(hs, x_mask)

        sentence = self.drop(sentence)
        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)

        if not return_alpha:
            return logits

        return logits, _alpha
