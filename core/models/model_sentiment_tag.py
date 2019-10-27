# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from core.nn.encoder import TokenEmbedding, BiLstmEncoder
from core.nn.encoder import AveragePooling, MaxPooling, SelfAttention


class BilstmClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_dim,
                 lstm_dim,
                 n_sentiment_tag,
                 mlp_dim,
                 n_class,
                 we,
                 interaction_type):
        super(BilstmClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.n_sentiment_tag = n_sentiment_tag
        self.mlp_dim = mlp_dim
        self.n_class = n_class

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)
        self.bilstm = BiLstmEncoder(self.word_dim, self.lstm_dim)
        if interaction_type == "max":
            self.interaction_layer = MaxPooling()
        elif interaction_type == "avg":
            self.interaction_layer = AveragePooling()
        else:
            self.interaction_layer = SelfAttention(2 * self.lstm_dim, self.lstm_dim)

        self.mlp_layer = nn.Linear(2 * self.lstm_dim, self.mlp_dim)
        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

        self.word2sentiment_tag = nn.Linear(2*self.lstm_dim, self.n_sentiment_tag)

    def forward(self, x, x_len, x_mask):
        x_embedding = self.word_embedding(x)

        hs = self.bilstm(x_embedding, x_len)  # [B, T, 2*lstm_dim]

        sentence = self.interaction_layer(hs, x_mask)

        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)  # [B, n_class]

        hs_flatten = hs.view(-1, hs.shape[2])  # dim: [B*T x num_tags]
        tags_logits = self.word2sentiment_tag(hs_flatten)

        return logits, tags_logits
