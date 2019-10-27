# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class ClassifyCrossEntropy(nn.Module):
    def __init__(self):
        super(ClassifyCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, y):
        return self.criterion(logits, y)


class SentimentLoss(nn.Module):
    def __init__(self, n_sentiment_tag):
        super(SentimentLoss, self).__init__()
        self.n_sentiment_tag = n_sentiment_tag

    @staticmethod
    def tag_loss_fn(logits, labels):
        """
        Args:
            logits: 2D tensor, [B*T, n_tag]
            labels: 2D tensor, [B, T]

        Returns:

        """

        labels = labels.view(-1)  # [B*T]

        # mask out 'PAD' label, -1
        mask = (labels != -1).float()

        # the number of tokens is the sum of elements in mask
        num_tokens = mask.sum()

        # pick the values corresponding to labels and multiply by mask
        score = F.log_softmax(logits, dim=1)  # [B*T, n_tag]
        score_masked = score[range(score.shape[0]), labels] * mask

        # cross entropy loss for all non 'PAD' tokens
        return -score_masked.sum() / num_tokens

    def forward(self, words_logits, words_label, logits, y):
        """
        Args:
            words_logits: 2D tensor, [B*T, n_tag]
            words_label: 2D tensor, [B, T]
            logits: 2D tensor, [B, n_class]
            y: 1D tensor, [B, ]

        Returns:

        """

        loss = F.cross_entropy(logits, y)
        sentiment_tag_loss = SentimentLoss.tag_loss_fn(words_logits, words_label)
        total_loss = loss + sentiment_tag_loss

        return total_loss
