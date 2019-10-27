# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from core.nn.encoder import TokenEmbedding, BiLstmEncoder
from core.nn.encoder import SelfAttention

__all__ = ['BiLSTMWordAttentionAndTaging']


class BiLSTMWordAttentionAndTaging(nn.Module):
    """

    use bilstm to model word_embedding

    and Auxiliary with tagging task
    """
    def __init__(self,
                 vocab_size,
                 word_dim,
                 n_sentiment_tag,
                 tag_dim,
                 lstm_dim,
                 mlp_dim,
                 n_class,
                 we=None):
        super(BiLSTMWordAttentionAndTaging, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.n_sentiment_tag = n_sentiment_tag
        self.tag_dim = tag_dim

        self.lstm_dim = lstm_dim

        assert self.lstm_dim == self.word_dim,  "lstm_dim != word_dim"

        self.mlp_dim = mlp_dim
        self.n_class = n_class
        self.drop = nn.Dropout(p=0.5)

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)

        self.bilstm = BiLstmEncoder(self.lstm_dim, self.lstm_dim)
        self.interaction_layer = SelfAttention(2 * self.lstm_dim, self.lstm_dim)

        self.mlp_layer = nn.Linear(2 * self.lstm_dim, self.mlp_dim)

        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

        self.h2sentiment_tag = nn.Linear(2 * self.lstm_dim, self.n_sentiment_tag)

    def forward(self, word_inputs, tag_inputs, seq_lengths, mask):
        """
        Args:
            word_inputs: 2D tensor, [B, T]
            tag_inputs: 2D tensor, [B, T]
            seq_lengths: 1D tensor, [B]
            mask: 2D tensor, [B, T]

        Returns:

        """
        word_embedding = self.word_embedding(word_inputs)   # [B, T, word_dim]
        x_embedding = self.drop(word_embedding)

        hs = self.bilstm(x_embedding, seq_lengths)  # [B, T, 2*lstm_dim]

        sentence, alpha = self.interaction_layer(hs, mask)

        sentence = self.drop(sentence)
        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)  # [B, n_class]

        hs_flatten = hs.view(-1, hs.shape[2])  # dim: [B*T x D] , D = 2*lstm_dim

        tags_logits = self.h2sentiment_tag(hs_flatten)  # dim: [B*T x n_sentiment_tag]

        return logits, tags_logits

    def get_attention(self, word_inputs, seq_lengths, mask):
        """
        Args:
            word_inputs: 2D tensor, [B, T]
            seq_lengths: 1D tensor, [B]
            mask: 2D tensor, [B, T]

        Returns:

        """
        word_embedding = self.word_embedding(word_inputs)  # [B, T, word_dim]
        hs = self.bilstm(word_embedding, seq_lengths)  # [B, T, 2*lstm_dim]

        _, alpha = self.interaction_layer(hs, mask)

        return alpha
