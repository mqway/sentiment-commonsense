# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs


def load_sentiment_lexicon(file_positive, file_negative, file_negation, file_intensifier):

    sentiment_lexicon = {}

    def load(in_file):
        word_list = []
        with codecs.open(in_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                w = " ".join(line.strip().split('-'))
                word_list.append(w)
        return word_list

    sentiment_lexicon['positive'] = load(file_positive)
    sentiment_lexicon['negative'] = load(file_negative)
    sentiment_lexicon['negation'] = load(file_negation)
    sentiment_lexicon['intensifier'] = load(file_intensifier)

    return sentiment_lexicon
