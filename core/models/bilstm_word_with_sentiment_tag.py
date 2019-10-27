# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from core.nn.encoder import TokenEmbedding, BiLstmEncoder
from core.nn.encoder import SelfAttention

__all__ = ['BiLSTMWordTag', 'BiLSTMWordBiLSTMTag', 'BiLSTMWordTagBiLSTM']


class BiLSTMWordTag(nn.Module):
    """
    cat([word_embedding, tag_embedding], 2)

    and use bilstm to model word_embedding + tag_embedding
    """
    def __init__(self,
                 vocab_size,
                 word_dim,
                 n_sentiment_tag,
                 tag_dim,
                 lstm_dim,
                 mlp_dim,
                 n_class,
                 we):
        super(BiLSTMWordTag, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.n_sentiment_tag = n_sentiment_tag
        self.tag_dim = tag_dim

        self.lstm_dim = lstm_dim

        assert self.lstm_dim == self.word_dim + self.tag_dim, "lstm_dim != word_dim + tag_dim"

        self.mlp_dim = mlp_dim
        self.n_class = n_class
        self.drop = nn.Dropout(p=0.5)

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)
        self.tag_embedding = nn.Embedding(self.n_sentiment_tag + 1, self.tag_dim, padding_idx=0)

        self.bilstm = BiLstmEncoder(self.lstm_dim, self.lstm_dim)
        self.interaction_layer = SelfAttention(2 * self.lstm_dim, self.lstm_dim)

        self.mlp_layer = nn.Linear(2 * self.lstm_dim, self.mlp_dim)

        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

    def forward(self, word_inputs, tag_inputs, seq_lengths, mask):
        """
        Args:
            word_inputs: 2D tensor, [B, T]
            tag_inputs: 2D tensor, [B, T]
            seq_lengths: 1D tensor, [B]
            mask: 2D tensor, [B, T]

        Returns:

        """
        word_embedding = self.word_embedding(word_inputs)

        tag_inputs = tag_inputs + 1
        tag_embedding = self.tag_embedding(tag_inputs)

        x_embedding = torch.cat([word_embedding, tag_embedding], 2)  # [B, T, 2*lstm_dim]
        x_embedding = self.drop(x_embedding)

        hs = self.bilstm(x_embedding, seq_lengths)  # [B, T, 2*lstm_dim]

        sentence, alpha = self.interaction_layer(hs, mask)

        sentence = self.drop(sentence)
        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)  # [B, n_class]

        return logits


class BiLSTMWordBiLSTMTag(nn.Module):
    """
    use bilstm1 to model word_seq
    use bilstm2 to model tag_seq

    and cat([hs_word, hs_tag], 2)
    """
    def __init__(self,
                 vocab_size,
                 word_dim,
                 n_sentiment_tag,
                 tag_dim,
                 word_lstm_dim,
                 mlp_dim,
                 n_class,
                 we=None):
        super(BiLSTMWordBiLSTMTag, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.word_lstm_dim = word_lstm_dim

        self.n_sentiment_tag = n_sentiment_tag
        self.tag_dim = tag_dim
        self.lstm_tag_dim = 50

        self.mlp_dim = mlp_dim
        self.n_class = n_class
        self.drop = nn.Dropout(p=0.5)

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)
        self.word_bilstm = BiLstmEncoder(self.word_dim, self.word_lstm_dim)

        self.tag_embedding = nn.Embedding(self.n_sentiment_tag + 1, self.tag_dim, padding_idx=0)
        self.tag_bilstm = BiLstmEncoder(self.tag_dim, self.lstm_tag_dim)

        self.interaction_layer = SelfAttention(2 * self.word_lstm_dim + 2 * self.lstm_tag_dim, self.word_lstm_dim + self.lstm_tag_dim)

        self.mlp_layer = nn.Linear(2 * self.word_lstm_dim + 2 * self.lstm_tag_dim, self.mlp_dim)

        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

    def forward(self, word_inputs, tag_inputs, seq_lengths, mask):
        """
        Args:
            word_inputs: 2D tensor, [B, T]
            tag_inputs: 2D tensor, [B, T]
            seq_lengths: 1D tensor, [B]
            mask: 2D tensor, [B, T]

        Returns:

        """
        word_embedding = self.word_embedding(word_inputs)
        word_embedding = self.drop(word_embedding)
        hs_word = self.word_bilstm(word_embedding, seq_lengths)  # [B, T, 2*lstm_dim]

        tag_inputs = tag_inputs + 1
        tag_embedding = self.tag_embedding(tag_inputs)
        tag_embedding = self.drop(tag_embedding)
        hs_tag = self.tag_bilstm(tag_embedding, seq_lengths)  # [B, T, 2*lstm_tag_dim]

        hs = torch.cat([hs_word, hs_tag], 2)  # [B, T, 2*lstm_dim+2*lstm_tag_dim]

        sentence, alpha = self.interaction_layer(hs, mask)

        sentence = self.drop(sentence)
        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)  # [B, n_class]

        return logits


class BiLSTMWordTagBiLSTM(nn.Module):
    """
        use bilstm to model tag_seq
        and cat([word_embedding, hs_tag], 2)

        last use bilstm2 to model words + tag_embedding
        """
    def __init__(self,
                 vocab_size,
                 word_dim,
                 n_sentiment_tag,
                 tag_dim,
                 lstm_tag_dim,
                 lstm_dim,
                 mlp_dim,
                 n_class,
                 we):
        super(BiLSTMWordTagBiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim

        self.n_sentiment_tag = n_sentiment_tag
        self.tag_dim = tag_dim
        self.lstm_tag_dim = lstm_tag_dim

        self.mlp_dim = mlp_dim
        self.n_class = n_class
        self.drop = nn.Dropout(p=0.5)

        self.word_embedding = TokenEmbedding(self.vocab_size, self.word_dim, pretrained_emb=we)

        self.tag_embedding = nn.Embedding(self.n_sentiment_tag + 1, self.tag_dim, padding_idx=0)
        self.tag_bilstm = BiLstmEncoder(self.tag_dim, self.lstm_tag_dim)

        # lstm_dim = word_dim + 2*lstm_tag_dim
        assert self.lstm_dim > 2*self.lstm_tag_dim

        self.bilstm = BiLstmEncoder(self.lstm_dim, self.lstm_dim)

        self.interaction_layer = SelfAttention(2*self.lstm_dim, self.lstm_dim)

        self.mlp_layer = nn.Linear(2*self.lstm_dim, self.mlp_dim)

        self.classify_layer = nn.Linear(self.mlp_dim, self.n_class)

    def forward(self, word_inputs, tag_inputs, seq_lengths, mask):
        """
        Args:
            word_inputs: 2D tensor, [B, T]
            tag_inputs: 2D tensor, [B, T]
            seq_lengths: 1D tensor, [B]
            mask: 2D tensor, [B, T]

        Returns:

        """
        word_embedding = self.word_embedding(word_inputs)
        word_embedding = self.drop(word_embedding)  # [B, T, word_dim]

        tag_inputs = tag_inputs + 1
        tag_embedding = self.tag_embedding(tag_inputs)
        tag_embedding = self.drop(tag_embedding)

        hs_tag = self.tag_bilstm(tag_embedding, seq_lengths)  # [B, T, 2*lstm_tag_dim]

        x_embedding = torch.cat([word_embedding, hs_tag], 2)  # [B, T, word_dim+2*lstm_tag_dim]

        hs = self.bilstm(x_embedding, seq_lengths)  # [B, T, 2*lstm_dim]

        sentence, alpha = self.interaction_layer(hs, mask)

        sentence = self.drop(sentence)
        sentence = self.mlp_layer(sentence)

        logits = self.classify_layer(sentence)  # [B, n_class]

        return logits
