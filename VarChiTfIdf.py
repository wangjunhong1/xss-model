import re
import jieba
import math
import numpy as np
from tqdm import tqdm
import logging

jieba.setLogLevel(logging.INFO)


class VarChiTfIdf(object):
    def __init__(self):
        self.word_list = None
        self.vocabulary = None
        self.idf_list = None
        self.tf_idf_ = None
        self.csai = None
        self.split_index_list = None

    def index_word(self):
        return {v[1]: v[0] for v in self.vocabulary.items()}

    # 分词
    def word_tokenizer(self,
                       documents,
                       stop_word_pattern=
                       """([\\\`~!@#$%^&*()_=\[{\]};+\-:'"<,>.?/|]{1,})|(^\d*$)|(^\s*$)|(^�*$)"""):
        documents = [str(_).lower() for _ in documents]
        word_list = []
        for document in tqdm(documents, desc='%-30s' % 'word tokenizing'):
            word_list.append(jieba.lcut_for_search(document))
        self.word_list = []
        for _list in word_list:
            temp = []
            for word in _list:
                if re.match(stop_word_pattern, word) is None:
                    temp.append(str(word))
            self.word_list.append(temp)
        self.word_list = np.array(self.word_list, dtype=object)
        return self.word_list

    # 根据分词结果得到词典
    def dictionary(self, dictionary_size):
        # 统计所有单词数
        counter = {}
        for _list in tqdm(self.word_list, desc='%-30s' % 'building dictionary'):
            for word in _list:
                if word in counter:
                    counter[word] += 1
                else:
                    counter[word] = 1
        # 按照词的出现次数进行排序
        counter_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        dictionary_size = min(dictionary_size, len(counter_items))
        self.vocabulary = {counter_items[i][0]: i for i in range(dictionary_size)}
        return self.vocabulary

    # 计算TF
    def tf(self, documents):
        tf_list = []
        word_list = self.word_tokenizer(documents)
        if self.vocabulary is None:
            return None
        for _list in tqdm(word_list, desc='%-30s' % 'calculating tf'):
            _tf = {word: 0 for word in self.vocabulary}
            for word in _list:
                if word not in self.vocabulary:
                    pass
                if word in _tf:
                    _tf[word] += 1 / len(_list)
            tf_list.append([float(it) for it in _tf.values()])
        tf_list = np.array(tf_list)
        return tf_list

    # 计算IDF
    def idf(self):
        D = len(self.word_list)
        _idf = {}
        for word in tqdm(self.vocabulary, desc='%-30s' % 'calculating idf'):
            cnt = 0
            for _list in self.word_list:
                if word in _list:
                    cnt += 1
            _idf[word] = math.log(D / (1 + cnt))
        self.idf_list = [float(it) for it in _idf.values()]
        self.idf_list = np.array(self.idf_list)
        return self.idf_list

    def class_variance(self):
        dfi_dw = []
        split_index_list = [0] + self.split_index_list + [len(self.word_list)]
        for i in range(len(split_index_list) - 1):
            count = {word: 0 for word in self.vocabulary}
            for _list in self.word_list[split_index_list[i]:split_index_list[i + 1]]:
                # 统计_list 出现的单词数
                for word in _list:
                    if word in count:
                        count[word] += 1
            dfi_dw.append([count[word] for word in count.keys()])
        df_dw = np.reshape(np.sum(dfi_dw, axis=0), (-1))
        # 计算类频方差
        N = len(dfi_dw)
        fenzi = np.zeros((df_dw.shape))
        for i in range(N):
            fenzi += (df_dw / N - dfi_dw[i]) * (df_dw / N - dfi_dw[i])
        self.csai = np.sqrt(fenzi) / N
        self.csai = np.array(self.csai)
        return self.csai

    def fit(self, documents, split_index_list=[], dictionary_size=1000):
        self.split_index_list = split_index_list
        self.word_tokenizer(documents)
        self.dictionary(dictionary_size)
        self.idf()
        self.class_variance()
        return self

    def transform_imp_tf_idf(self, documents):
        tf_list = np.array(self.tf(documents))
        if self.idf_list is None:
            return None
        _idf = np.array(self.idf_list)
        if self.csai is None:
            return None
        self.tf_idf_ = np.array(tf_list * _idf * self.csai)
        return self.tf_idf_

    def transform_tf_idf(self, documents):
        tf_list = np.array(self.tf(documents))
        if self.idf_list is None:
            return None
        _idf = np.array(self.idf_list)
        self.tf_idf_ = np.array(tf_list * _idf)
        return self.tf_idf_

    def top_word_list(self, top=5):
        top = min(top, len(self.vocabulary))
        sorted_index_tfidf = []
        for _ in tqdm(self.tf_idf_, desc='%-30s' % 'getting sorted index_tfidf'):
            t = sorted(zip(range(len(_)), _), key=lambda x: x[1], reverse=True)
            sorted_index_tfidf.append(t)
        index_word = self.index_word()
        top_words = []
        for _list in tqdm(sorted_index_tfidf, desc='%-30s' % 'building top_words'):
            temp_list = []
            for i in range(top):
                if _list[i][1] > 0:
                    temp_list.append(index_word.get(_list[i][0]))
            top_words.append(temp_list)
        # top_words = np.array(top_words)
        return top_words
