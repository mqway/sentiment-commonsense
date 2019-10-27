# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs

g_tag_pad = -1


class SentimentLexicon:

    def __init__(self):
        self.lexicon = {}
        self._pad = g_tag_pad

    def load(self, f_negative, f_positive, f_negation, f_intensity):

        with codecs.open(f_negative, 'r', encoding='utf-8') as fr:
            for word in fr:
                self.lexicon[word.strip()] = 1

        with codecs.open(f_positive, 'r', encoding='utf-8') as fr:
            for word in fr:
                self.lexicon[word.strip()] = 2

        with codecs.open(f_negation, 'r', encoding='utf-8') as fr:
            for word in fr:
                self.lexicon[word.strip()] = 3

        with codecs.open(f_intensity, 'r', encoding='utf-8') as fr:
            for word in fr:
                self.lexicon[word.strip()] = 4

    def word_to_id(self, key):
        # default 3 for neutral
        return self.lexicon.get(key, 0)

    def encode(self, sentence, max_len=50):
        word_ids = [self.word_to_id(w) for w in sentence.split()][:max_len]
        seq_len = len(word_ids)
        word_ids = word_ids + [g_tag_pad] * (max_len - seq_len)
        return word_ids, seq_len

    @property
    def size(self):
        return 5

    @property
    def pad(self):
        return self._pad
