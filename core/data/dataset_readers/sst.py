# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs


def sst_binary_loader(in_file, preprocess=None):
    """
          c  l   N   Test
    SST-2 2 19 9613  1821
    //three sets having(train/dev/test) 6920/872/1821 sentences
    Args:
        in_file: str, eg, "sst.binary.train"
        preprocess: None, or function

    Returns: data_train

    """
    label2y = {'0': 0, '1': 1}

    y = []
    x = []
    with codecs.open(in_file, 'r', encoding="utf-8") as fr:
        for line in fr:
            t = line.strip().split("\t", maxsplit=1)
            assert len(t) == 2, "Wrong sst binary line format `label:sentence`"

            y.append(label2y[t[0]])
            sentence = t[1]
            if preprocess is not None:
                sentence = preprocess(sentence)
            x.append(sentence)

    return x, y


def sst_fine_loader(in_file, preprocess=None):
    """
    //three sets having(train/dev/test) 8544/1101/2210 sentences
    Args:
        in_file: str, eg, "sst.fine.train"
        preprocess: None, or function
    Returns: data_train

    """
    label2y = {str(i): i for i in range(5)}

    y = []
    x = []
    with codecs.open(in_file, 'r', encoding="utf-8") as fr:
        for line in fr:
            t = line.strip().split("\t", maxsplit=1)
            assert len(t) == 2, "Wrong sst binary line format `label:sentence`"

            y.append(label2y[t[0]])
            sentence = t[1]

            if preprocess is not None:
                sentence = preprocess(sentence)
            x.append(sentence)

    return x, y
