# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
from collections import Counter


def build_vocabulary_from_files(in_files, out_file, min_cnt=2, max_num=100000):

    counter = Counter()

    for in_file in in_files:
        with codecs.open(in_file, 'r', encoding='utf-8') as fr:
            for line in fr:

                t = line.strip().split("\t", maxsplit=1)
                if len(t) != 2:
                    print("len(t) != 2:", in_file)
                    print(line)
                    continue
                label, sentence = t
                sentence = sentence.replace(" -lrb- ", " ( ").replace(" -rrb- ", " ) ")
                sentence = sentence.replace('-', ' ')
                counter.update(sentence.split())

    words = counter.most_common(max_num)
    words = [(w, cnt) for (w, cnt) in words if cnt > min_cnt-1]
    # 4 for : pad, unk, bos, eos
    print("vocabulary total cnt: %d, max_num: %d" % (len(counter) + 4, max_num))
    print("vocabulary cnt (>min_cnt): %d, raw_cnt: %d" % (len(words) + 4, len(counter)))

    with codecs.open(out_file, 'w', encoding='utf-8') as fw:
        fw.write("<pad>\t1\n")
        fw.write("<unk>\t1\n")
        fw.write("<bos>\t1\n")
        fw.write("<eos>\t1\n")
        for w, cnt in words:
            fw.write("%s\t%d\n" % (w, cnt))


def oov_statistic(train_file, valid_file, test_file, embedding_file):

    tokens_train = set()
    with codecs.open(train_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            label, sentence = line.strip().split("\t", maxsplit=1)
            tokens_train.update(sentence.split())

    tokens_valid = set()
    with codecs.open(valid_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            label, sentence = line.strip().split("\t", maxsplit=1)
            tokens_valid.update(sentence.split())

    tokens_test = set()
    with codecs.open(test_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            label, sentence = line.strip().split("\t", maxsplit=1)
            tokens_test.update(sentence.split())

    big_tokens = set()
    with codecs.open(embedding_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            token = line.split(' ', maxsplit=1)[0]
            big_tokens.add(token)

    def count_oov(source, target):
        cnt = 0
        for t in source:
            if t not in target:
                cnt += 1
        return cnt

    oov_valid = count_oov(tokens_valid, tokens_train)
    oov_test = count_oov(tokens_test, tokens_train)
    print("[In training file] ==> miss_valid: %d, size: %d" % (oov_valid, len(tokens_valid)))
    print("[In training file] ==> miss_test: %d, size: %d" % (oov_test, len(tokens_test)))

    oov_valid = count_oov(tokens_valid, big_tokens)
    oov_test = count_oov(tokens_test, big_tokens)
    print("[In big embedding] ==> miss_valid: %d, size: %d" % (oov_valid, len(big_tokens)))
    print("[In big embedding] ==> miss_test: %d, size: %d" % (oov_test, len(big_tokens)))


if __name__ == '__main__':
    # [In training file] == > miss_valid: 940, size: 4339
    # [In training file] == > miss_test: 1934, size: 7053
    # [In big embedding] == > miss_valid: 123, size: 2196017
    # [In big embedding] == > miss_test: 245, size: 2196017
    # vocabulary total cnt: 16292, max_num: 100000
    # vocabulary size: 13810 (cnt >= 2)
    pass
