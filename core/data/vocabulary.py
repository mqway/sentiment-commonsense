# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import pickle
import numpy as np
from collections import defaultdict

g_pad = "<pad>"
g_unk = "<unk>"
g_bos = "<bos>"  # begin of sentence
g_eos = "<eos>"  # end of sentence


def save_params(params, out_file):
    if os.path.exists(out_file):
        os.remove(out_file)

    with codecs.open(out_file, 'wb') as fw:
        pickle.dump(params, fw, protocol=2)


def load_params(load_file):
    if not os.path.exists(load_file):
        raise RuntimeError('no file: %s' % load_file)
    with codecs.open(load_file, 'rb') as fr:
        params = pickle.load(fr)
    return params


def load_tokens(in_file: str, check_special_token=True):
    """
    Args:
        in_file: file path
        check_special_token: boolean

    Returns: dict, list

    """
    token2id = defaultdict()
    id2token = list()

    assert os.path.isfile(in_file)

    with codecs.open(in_file, 'r', encoding='utf-8') as fr:
        if check_special_token:
            assert fr.readline().strip().split("\t")[0] == g_pad
            assert fr.readline().strip().split("\t")[0] == g_unk
            assert fr.readline().strip().split("\t")[0] == g_bos
            assert fr.readline().strip().split("\t")[0] == g_eos

        for i, line in enumerate(fr):
            w = line.strip().split("\t")[0]
            token2id[w] = i
            id2token.append(w)
    return token2id, id2token


def save_tokens(out_file, id2token):
    """
    Args:
        out_file: file path
        id2token: words list

    Returns: None

    """
    with codecs.open(out_file, 'w', encoding='utf-8') as fw:
        for w in id2token:
            fw.write(w + "\n")


def get_tokens_from_embedding_file(in_file, head_line_is_info=True):

    token2id = defaultdict()
    id2token = [g_pad, g_unk, g_bos, g_eos]
    token2id[g_pad] = 0  # 0 for <pad>
    token2id[g_unk] = 1  # 1 for <unk>
    token2id[g_bos] = 2  # 1 for <bos>
    token2id[g_eos] = 3  # 1 for <eos>
    idx = 4

    with codecs.open(in_file, 'r', encoding='utf-8') as fr:
        if head_line_is_info:
            n_vocab, dim = fr.readline().strip().split()
            print("token info: %s %s" % (n_vocab, dim))

        for line in fr:
            token = line.split(maxsplit=1)[0]
            token2id[token] = idx
            idx += 1
            id2token.append(token)

    return token2id, id2token


class Vocabulary(object):
    """
        A token vocabulary.  Holds a map from token to ids and provides
        a method for encoding text to a sequence of ids.
    """

    def __init__(self, filename, from_embedding_file, max_sentence_length=15):

        self._pad = 0
        self._unk = 1
        self._bos = 2
        self._eos = 3
        self._max_sentence_length = max_sentence_length

        self.word2id, self.id2word = Vocabulary.load(filename, from_embedding_file)

        self.word2embedding = None
        self.oov = 0

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self.id2word)

    def word_to_id(self, word):
        return self.word2id.get(word, self._unk)

    def id_to_word(self, idx):
        return self.id2word[idx]

    def decode(self, idxs):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(idx) for idx in idxs])

    def encode(self, sentence, split=True, max_len=50):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.
        """

        if split:
            word_ids = [self.word_to_id(w) for w in sentence.split()][-max_len:]
            seq_len = len(word_ids)
            word_ids = word_ids + [0] * (max_len-seq_len)
        else:
            word_ids = [self.word_to_id(w) for w in sentence][-max_len:]
            seq_len = len(word_ids)
            word_ids = word_ids + [0] * (max_len - seq_len)

        return word_ids, seq_len

    def get_token_embedding(self, token_dim, embedding_file, head_line_is_info=False):
        n_token = len(self.word2id)

        scale = np.sqrt(3.0 / token_dim)
        we = np.random.uniform(-scale, scale, (n_token, token_dim)).astype(dtype=np.float32)
        we[0] = np.zeros(token_dim)

        n_found = 0
        line_len = token_dim + 1
        with codecs.open(embedding_file, 'r', encoding="utf-8") as fr:
            if head_line_is_info:
                n_vocab, dim = fr.readline().strip().split()
                assert int(dim) == token_dim
                print("token info: %s %s" % (n_vocab, dim))
            for line in fr:
                sp = line.strip().split(' ')
                if len(sp) != line_len:
                    continue
                idx = self.word2id.get(sp[0], -1)
                if idx != -1:
                    we[idx] = [float(t) for t in sp[1:]]
                    n_found += 1

        self.word2embedding = we
        self.oov = n_token - n_found
        print("oov num : %d/%d = %0.4f" % (self.oov, n_token, self.oov / n_token))

    def save(self, out_file):
        save_tokens(out_file, self.id2word)

    def save_embedding(self, out_file):
        save_params(self.word2embedding, out_file)

    def load_embedding(self, load_file):
        self.word2embedding = load_params(load_file)

    @staticmethod
    def load(in_file, from_embedding_file=True):
        if from_embedding_file:
            return get_tokens_from_embedding_file(in_file)
        else:
            return load_tokens(in_file)
