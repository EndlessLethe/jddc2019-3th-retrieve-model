# coding=utf-8
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re
# import sklearn
from jieba_seg import JiebaSeg
import time
# import logging
from gensim import corpora, models, similarities
from data_loader import DataLoader

class Tfidf():
    def __init__(self, filepath_input = "../data/JDDC_100W训练数据集/训练数据集/chat_1per.txt"):
        self.filepath_input = filepath_input
        self.dictionary = None
        self.tfidf = None
        self.index = None
        self.data_chat = self.load_data()
        self.seg_jieba = JiebaSeg()

    def load_data(self):
        data_chat = pd.read_csv(self.filepath_input, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
        # logging.debug("Data size: " + str(data_chat.shape[0]))
        return data_chat


    def fit(self):
        corpus_bow = self.get_corpus_bow(self.data_chat)
        corpus_tfidf = self.get_corpus_tfidf(corpus_bow)
        self.get_index(corpus_tfidf)

    def get_corpus_bow(self, data_chat):
        time_start=time.time()

        texts = []

        for i in range(data_chat.shape[0]):
            sentence = data_chat.iat[i, 6]
            list_word = list(self.seg_jieba.cut(sentence, True))
            texts.append(list_word)
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=100000, keep_tokens=None)
        corpus_bow = [self.dictionary.doc2bow(text) for text in texts]

        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        return corpus_bow

    def get_corpus_tfidf(self, corpus_bow):
        time_start=time.time()

        self.tfidf = models.TfidfModel(corpus_bow)
        corpus_tfidf = self.tfidf[corpus_bow]

        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        return corpus_tfidf

    def get_index(self, corpus_tfidf):
        time_start=time.time()

        # self.index = similarities.MatrixSimilarity(corpus_tfidf, num_features=len(self.dictionary))
        self.index = similarities.Similarity("out/", corpus_tfidf, len(self.dictionary))

        # with open("../data/JDDC_100W训练数据集/训练数据集/corpus_bow.pkl", 'w') as f:
        #     pickle.dump([dictionary, corpus_bow], f)
        # tfidf.save("../data/JDDC_100W训练数据集/训练数据集/tfidf.model")
        # index.save("../data/JDDC_100W训练数据集/训练数据集/index.index")

        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        return self.index

    ## 先使用sentence2vec将需要匹配的句子传进去
    def sentence2vec(self, sentence):
        list_word = list(self.seg_jieba.cut(sentence))
        vec_bow = self.dictionary.doc2bow(list_word)
        return self.tfidf[vec_bow]

    def get_topk_answer(self, sentence, k = 15):
        """求最相似的句子"""
        sentence_vec = self.sentence2vec(sentence)
        sims = self.index[sentence_vec]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_k = sim_sort[0:k]

        return top_k

    def get_word_vector(self, s1,s2):
        """
        :param s1: 句子1
        :param s2: 句子2
        :return: 返回句子的余弦相似度
        """
        # 分词
        cut1 = self.seg_jieba.cut(s1)
        cut2 = self.seg_jieba.cut(s2)
        list_word1 = (','.join(cut1)).split(',')
        list_word2 = (','.join(cut2)).split(',')

        # 列出所有的词,取并集
        key_word = list(set(list_word1 + list_word2))
        # 给定形状和类型的用0填充的矩阵存储向量
        word_vector1 = np.zeros(len(key_word))
        word_vector2 = np.zeros(len(key_word))

        # 计算词频
        # 依次确定向量的每个位置的值
        for i in range(len(key_word)):
            # 遍历key_word中每个词在句子中的出现次数
            for j in range(len(list_word1)):
                if key_word[i] == list_word1[j]:
                    word_vector1[i] += 1
            for k in range(len(list_word2)):
                if key_word[i] == list_word2[k]:
                    word_vector2[i] += 1

        # 输出向量
    #     print(word_vector1)
    #     print(word_vector2)
        return word_vector1, word_vector2

    def cos_dist(self, s1, s2):
        """
        :param vec1: 向量1
        :param vec2: 向量2
        :return: 返回两个向量的余弦相似度
        """
        vec1, vec2 = self.get_word_vector(s1, s2)
        dist1= float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        return dist1

    def get_center_sentence(self, list_list_kanswer, k):
        x_sim = np.zeros((k, k))
        for i in range(k):
            if len(self.data_chat.iat[list_list_kanswer[i][0] + 1, 6]) <= 5: continue
            for j in range(k):
                if i != j:
                    x_sim[i][j] = self.cos_dist(self.data_chat.iat[list_list_kanswer[i][0] + 1, 6],
                                                self.data_chat.iat[list_list_kanswer[j][0] + 1, 6])
        #             print("i j: " + str(i) + " " + str(j) + " " + str(cos_dist(data_chat.iat[list_list_kanswer[i][0]+1, 6], data_chat.iat[list_list_kanswer[j][0]+1, 6])))

        # print(x_sim)

        x_sum = np.zeros((k,))
        for i in range(k):
            x_sum[i] = x_sim[i].sum()
        n_result = np.argmax(x_sum)
        return n_result

    def get_maxlen_sentence(self, list_list_kanswer, k):
        x_len = np.zeros((k, ))
        for i in range(k):
            x_len[i] = len(self.data_chat.iat[list_list_kanswer[i][0] + 1, 6])
        n_result = np.argmax(x_len)
        return n_result


    def similarity(self, sentence, k = 10):
        list_list_kanswer = self.get_topk_answer(sentence, k)
        # n_result = self.get_center_sentence(list_list_kanswer, k)
        n_result = self.get_maxlen_sentence(list_list_kanswer, k)

        return n_result, self.data_chat.iat[list_list_kanswer[n_result][0]+1, 6]

    def predict(self, session_list, session_length, session_text, filepath_result):
        time_start = time.time()
        with open(filepath_result, "w", encoding='utf-8') as f_out:
            cnt = 0
            for i in range(len(session_list)):
                f_out.write("<session " + session_list[i] + ">\n")
                for j in range(session_length[i]):
                    f_out.write(self.similarity(session_text[cnt])[1] + "\n")
                    cnt += 1
                f_out.write("</session " + session_list[i] + ">\n\n")

        time_end = time.time()
        print('time cost', time_end - time_start, 's')


# filepath_input = "../data/JDDC_100W训练数据集/训练数据集/chat_1per.txt"
# filepath_result = "../output/ans.txt"
# model_tfidf = Tfidf(filepath_input, filepath_result)
# model_tfidf.fit()
#
# data_loader = DataLoader()
# session_list, session_length, session_text = data_loader.read_file()
# model_tfidf.predict(session_list, session_length, session_text)
# print("Finish")