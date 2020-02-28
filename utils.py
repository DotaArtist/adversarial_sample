#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方脚本"""

__author__ = 'yp'

import re
import mafan
import pandas as pd
import numpy as np
import editdistance
from tqdm import tqdm
from sklearn.utils import shuffle
from gensim.models import KeyedVectors


# import fasttext
# def reference_model(model_path='./mini.ftz', test_args=('你好',)):
#     """
#     调用参考模型; 需要fasttext
#     """
#     model = fasttext.load_model(model_path)
#     print('----Reference model prediction----')
#     print(*test_args, model.predict(*test_args))


EMBEDDING_PATH = './zh.300.vec.gz'
EMBEDDING_DIM = 300


# 预处理模块
REGEX_TO_REMOVE = re.compile(
    r"[^\u4E00-\u9FA5a-zA-Z0-9\!\@\#\$\%\^\&\*\(\)\-\_\+\=\`\~\\\|\[\]\{\}\:\;\"\'\,\<\.\>\/\?\ \t，。！？]")


def preprocess_text(text, truncate_at=100):
    '''
    清除限定字符集外的内容，并截取前若干字符的文本.
    '''
    truncated = text[:truncate_at]
    cleaned = REGEX_TO_REMOVE.sub(r'', truncated)

    return mafan.simplify(cleaned)


def distance_measure(test_args=(['你好呀'], ['你好'])):
    """
    调用距离计算器; 需要gensim, numpy
    """
    dc = DistanceCalculator()
    print('----Distance measure----')
    print(*test_args, dc(*test_args))


def preprocess_example(test_args=('慶曆四年春，滕（téng）子京謫（zhé）守巴陵郡。越明年，政通人和，百廢俱興。乃 重修岳陽樓，增其舊制，刻唐賢、今人詩賦於其上。',)):
    """
    调用评测中使用的预处理函数; 需要mafan
    """
    print('----Text preprocessing----')
    print(*test_args, preprocess_text(*test_args))


# 相似度计算模块
def normalized_levenshtein(str_a, str_b):
    '''
    Edit distance normalized to [0, 1].
    '''
    return min(editdistance.eval(str_a, str_b) / (len(str_b) + 1e-16), 1.0)


def jaccard_set(set_a, set_b):
    '''
    Jaccard SIMILARITY between sets.
    '''
    set_c = set_a.intersection(set_b)
    return float(len(set_c)) / (len(set_a) + len(set_b) - len(set_c) + 1e-16)


def jaccard_char(str_a, str_b):
    '''
    Jaccard DISTANCE between strings, evaluated by characters.
    '''
    set_a = set(str_a)
    set_b = set(str_b)
    return 1.0 - jaccard_set(set_a, set_b)


def jaccard_word(str_a, str_b, sep=' '):
    '''
    Jaccard DISTANCE between strings, evaluated by words.
    '''
    set_a = set(str_a.split(sep))
    set_b = set(str_b.split(sep))
    return 1.0 - jaccard_set(set_a, set_b)


# 词向量读取模块
def tokenize(text):
    import jieba
    return ' '.join(jieba.cut(text))


class DistanceCalculator:
    '''
    Computes pair-wise distances between texts, using multiple metrics.
    '''

    def __init__(self):
        pass

    def __call__(self, docs_a, docs_b):
        docs_a_cut = [tokenize(_doc) for _doc in docs_a]
        docs_b_cut = [tokenize(_doc) for _doc in docs_b]

        # further validating input
        if not self.validate_input(docs_a, docs_b):
            raise ValueError("distance module got invalid input")

        # actual processing
        num_elements = len(docs_a)
        distances = dict()
        distances['normalized_levenshtein'] = [normalized_levenshtein(docs_a[i], docs_b[i]) for i in range(num_elements)]
        distances['jaccard_word'] = [jaccard_word(docs_a_cut[i], docs_b_cut[i]) for i in range(num_elements)]
        distances['jaccard_char'] = [jaccard_char(docs_a[i], docs_b[i]) for i in range(num_elements)]
        distances['embedding_cosine'] = self.batch_embedding_cosine_distance(docs_a_cut, docs_b_cut)
        return distances

    def validate_input(self, text_list_a, text_list_b):
        '''
        Determine whether two arguments are lists containing the same number of strings.
        '''
        if not (isinstance(text_list_a, list) and isinstance(text_list_b, list)):
            return False

        if not len(text_list_a) == len(text_list_b):
            return False

        for i in range(len(text_list_a)):
            if not (isinstance(text_list_a[i], str) and isinstance(text_list_b[i], str)):
                return False

        return True

    def batch_embedding_cosine_distance(self, text_list_a, text_list_b):
        '''
        Compute embedding cosine distances in batches.
        '''
        import numpy as np
        embedding_array_a = np.array(batch_doc2vec(text_list_a))
        embedding_array_b = np.array(batch_doc2vec(text_list_b))
        norm_a = np.linalg.norm(embedding_array_a, axis=1)
        norm_b = np.linalg.norm(embedding_array_b, axis=1)
        cosine_numer = np.multiply(embedding_array_a, embedding_array_b).sum(axis=1)
        cosine_denom = np.multiply(norm_a, norm_b)
        cosine_dist = 1.0 - np.divide(cosine_numer, cosine_denom)
        return cosine_dist.tolist()


class DataProcess(object):
    def __init__(self, _show_token=False):
        self.bert_batch_size = 32
        self.batch_size = 32
        self.data_path = None
        self.show_token = _show_token
        self.data = None
        self.data_x = None
        self.data_y = None
        self.sentence_data = None
        self.DEFAULT_KEYVEC = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, limit=50000)

    def doc2vec(self, tokenized):
        tokens = tokenized.split(' ')
        vec = np.full(EMBEDDING_DIM, 1e-10)
        weight = 1e-8
        for _token in tokens:
            try:
                vec += self.DEFAULT_KEYVEC.get_vector(_token)
                weight += 1.0
            except:
                pass
        return vec / weight

    def batch_doc2vec(self, list_of_tokenized_text):
        return [self.doc2vec(_text) for _text in list_of_tokenized_text]

    def load_data(self, file_list, is_shuffle=True):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:

            sentence_list = []
            label_list = []

            data_tmp = pd.DataFrame()
            with open(i, encoding='utf-8', mode='r') as _f:
                for line in _f.readlines():
                    sentence, label = line.strip().strip("\n").split('\t')

                    sentence_list.append(sentence)
                    label_list.append(label)

            data_tmp['sentence'] = pd.Series(sentence_list)
            data_tmp['label'] = pd.Series(label_list)

            data = pd.concat([data, data_tmp])

        if is_shuffle:
            data = shuffle(data)
        self.data = data

    def get_feature(self):
        data_x = []
        data_y = []
        sentence_data = []

        _sentence_list = []

        for index, row in tqdm(self.data.iterrows()):
            label = row['label']
            data_y.append(label)
            sentence_data.append(row['sentence'])

            _sentence = row['sentence']
            _sentence_list.append(_sentence)

            if len(_sentence_list) == 32:
                data_x.extend(batch_doc2vec(_sentence_list))
                _sentence_list = []

        if len(_sentence_list) > 0:
            data_x.extend(batch_doc2vec(_sentence_list))

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        sentence_data = np.array(sentence_data)

        self.data_x = data_x
        self.data_y = data_y
        self.sentence_data = sentence_data

        print("data_x shape:", data_x.shape)
        print("data_y shape:", data_y.shape)
        print("sentence_data shape:", sentence_data.shape)

    def next_batch(self):
        counter = 0
        batch_x = []
        batch_y = []
        batch_sen = []

        for (_x, _y, _sen) in zip(self.data_x, self.data_y, self.sentence_data):
            if counter == 0:
                batch_x = []
                batch_y = []
                batch_sen = []

            batch_x.append(_x)
            batch_y.append(_y)
            batch_sen.append(_sen)

            counter += 1

            if counter == self.batch_size:
                counter = 0
                yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)
        yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)

    def get_one_sentence_feature(self, sentence):
        data_x = []
        data_y = []
        data_x.extend(doc2vec(sentence))
        data_y.append([0])
        return np.array(data_x), np.array(data_y, dtype=np.int64)

    def get_sentence_list_feature(self, sentence_list):
        data_x = []
        data_y = []
        data_x.extend(batch_doc2vec(sentence_list))
        [data_y.append([0]) for i in sentence_list]
        return np.array(data_x), np.array(data_y, dtype=np.int64)


def segment_content_origin(content):
    _content_list = re.split("。", content)

    if len(_content_list) == 1:
        pass
    else:
        _content_list = [i + '。' for i in _content_list if i != ""]
    return _content_list


def segment_content(line, length=100):
    line = line.rstrip()
    if len(line) > length:
        ans = []
        sub_sentences = re.split('。', line)
        sub_sentences = [
            sub + '。' if idx != len(sub_sentences) - 1 else sub
            for idx, sub in enumerate(sub_sentences)
        ]
        new_line = ''
        for idx_1, sub_1 in enumerate(sub_sentences):
            # if len(sub_1) == 0:
            #     continue
            sub_1 = re.sub(' +', ' ', sub_1)
            if len(sub_1) <= length:
                if len(new_line) <= length and len(new_line + sub_1) >= length:
                    ans.append(new_line)
                    new_line = sub_1
                else:
                    new_line += sub_1
                if idx_1 == len(sub_sentences) - 1 and len(new_line) < length:
                    ans.append(new_line)
                    new_line = ''
            else:
                sub_1 = re.split('；', sub_1)
                sub_1 = [
                    sub + '；' if idx != len(sub_1) - 1 else sub
                    for idx, sub in enumerate(sub_1)
                ]

                sub_1_out = []
                for sub_1_deep in sub_1:
                    sub_1_deep = re.split(';', sub_1_deep)
                    sub_1_deep = [
                        sub + ';' if idx != len(sub_1_deep) - 1 else sub
                        for idx, sub in enumerate(sub_1_deep)
                    ]
                    sub_1_out.extend(sub_1_deep)

                sub_1 = sub_1_out

                for idx_2, sub_2 in enumerate(sub_1):
                    # if len(sub_2) == 0:
                    #     continue
                    if len(sub_2) <= length:
                        if len(new_line) <= length and len(new_line + sub_2) >= length:
                            ans.append(new_line)
                            new_line = sub_2
                        else:
                            new_line += sub_2
                        if idx_2 == len(sub_1) - 1 and len(new_line) < length:
                            ans.append(new_line)
                            new_line = ''
                    else:
                        sub_2 = re.split('，', sub_2)
                        sub_2 = [
                            sub + '，' if idx != len(sub_2) - 1 else sub
                            for idx, sub in enumerate(sub_2)
                        ]

                        sub_2_out = []
                        for sub_2_deep in sub_2:
                            sub_2_deep = re.split(',', sub_2_deep)
                            sub_2_deep = [
                                sub + ',' if idx != len(sub_2_deep) - 1 else sub
                                for idx, sub in enumerate(sub_2_deep)
                            ]
                            sub_2_out.extend(sub_2_deep)

                        sub_2 = sub_2_out

                        for idx_3, sub_3 in enumerate(sub_2):
                            # if len(sub_3) == 0:
                            #     continue
                            if len(sub_3) <= length:
                                if len(new_line) <= length and len(new_line + sub_3) >= length:
                                    ans.append(new_line)
                                    new_line = sub_3
                                else:
                                    new_line += sub_3
                                if idx_3 == len(sub_2) - 1 and len(new_line) < length:
                                    ans.append(new_line)
                                    new_line = ''

                            else:
                                sub_3 = re.split('、', sub_3)
                                sub_3 = [
                                    sub + '、' if idx != len(sub_3) - 1 else sub
                                    for idx, sub in enumerate(sub_3)
                                ]
                                for idx_4, sub_4 in enumerate(sub_3):
                                    # if len(sub_4) == 0:
                                    #     continue
                                    if len(sub_4) <= length:
                                        if len(new_line) <= length and len(new_line + sub_4) >= length:
                                            ans.append(new_line)
                                            new_line = sub_4
                                        else:
                                            new_line += sub_4
                                        if idx_4 == len(sub_3) - 1 and len(new_line) < length:
                                            ans.append(new_line)
                                            new_line = ''
                                    else:
                                        sub_4 = re.split(' ', sub_4)
                                        sub_4 = [
                                            sub + ' '
                                            if idx != len(sub_4) - 1 else sub
                                            for idx, sub in enumerate(sub_4)
                                        ]
                                        for idx_5, sub_5 in enumerate(sub_4):
                                            if len(sub_5) == 0:
                                                continue
                                            if len(sub_5) <= length:
                                                if len(new_line) <= length and len(new_line +sub_5) >= length:
                                                    ans.append(new_line)
                                                    new_line = sub_5
                                                else:
                                                    new_line += sub_5
                                                if idx_5 == len(sub_4) - 1 and len(new_line) < length:
                                                    ans.append(new_line)
                                                    new_line = ''
                                            else:
                                                ans.append(sub_5)
        ans_process = []
        for i in ans:
            _tmp = segment_content_origin(i)
            ans_process.extend(_tmp)
        return ans_process
    else:
        _tmp = segment_content_origin(line)
        return _tmp


if __name__ == '__main__':
    # reference_model()
    # distance_measure()
    # preprocess_example()
    with open('./maren.txt', encoding='utf-8', mode='r') as f1:
        for line in f1.readlines():
            __ = segment_content(line.strip())
            for i in __:
                print(i)
