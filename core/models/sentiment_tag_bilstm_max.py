# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from core.nn.encoder import TokenEmbedding, BiLstmEncoder
from core.nn.encoder import MaxPooling


class BilstmClassifier(nn.Module):
    def __init__(self,
                 n_sentiment_tag=5,
                 tag_dim=50,
                 lstm_dim=50,
                 mlp_dim=100,
                 n_class=2):

        super(BilstmClassifier, self).__init__()

        self.n_sentiment_tag = n_sentiment_tag
        self.tag_dim = tag_dim
        self.lstm_tag_dim = lstm_dim

        self.mlp_dim = mlp_dim
        self.n_class = n_class
        self.drop = nn.Dropout(p=0.5)

        self.tag_embedding = TokenEmbedding(1+self.n_sentiment_tag, self.tag_dim)

        self.tag_bilstm = BiLstmEncoder(self.tag_dim, self.lstm_tag_dim)

        self.interaction_layer = MaxPooling()

        self.mlp_layer = nn.Linear(2*self.lstm_tag_dim, self.mlp_dim)

        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

    def forward(self, tag_inputs, seq_lengths, mask):
        """
        Args:
            tag_inputs: 2D tensor, [B, T]
            seq_lengths: 1D tensor, [B]
            mask: 2D tensor, [B, T]

        Returns:

        """

        tag_inputs = tag_inputs + 1
        tag_embedding = self.tag_embedding(tag_inputs)
        tag_embedding = self.drop(tag_embedding)
        hs = self.tag_bilstm(tag_embedding, seq_lengths)  # [B, T, 2*lstm_tag_dim]

        sentence = self.interaction_layer(hs, mask)

        sentence = self.drop(sentence)
        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)  # [B, n_class]

        return logits
