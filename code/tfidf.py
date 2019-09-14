# coding=utf-8
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re
# import sklearn
import jieba
import time
import logging
from gensim import corpora, models, similarities
from data_loader import DataLoader

class Tfidf():
    def __init__(self, filename = "chat_1per.txt", filedir = "../data/JDDC_100W训练数据集/训练数据集/"):
        self.filedir = filedir
        self.filename = filename
        self.dictionary = None
        self.tfidf = None
        self.index = None
        self.data_chat = self.load_data()

    def load_data(self):
        filepath = os.path.join(self.filedir, self.filename)
        data_chat = pd.read_csv(filepath, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
        logging.debug("Data size: " + str(data_chat.shape[0]))
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
            list_word = list(jieba.cut(sentence))
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
        self.index = similarities.Similarity("../output/", corpus_tfidf, len(self.dictionary))

        # with open("../data/JDDC_100W训练数据集/训练数据集/corpus_bow.pkl", 'w') as f:
        #     pickle.dump([dictionary, corpus_bow], f)
        # tfidf.save("../data/JDDC_100W训练数据集/训练数据集/tfidf.model")
        # index.save("../data/JDDC_100W训练数据集/训练数据集/index.index")

        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        return self.index

    ## 先使用sentence2vec将需要匹配的句子传进去
    def sentence2vec(self, sentence):
        list_word = list(jieba.cut(sentence))
        vec_bow = self.dictionary.doc2bow(list_word)
        return self.tfidf[vec_bow]

    def get_topk_answer(self, sentence, k = 15):
        """求最相似的句子"""
        sentence_vec = self.sentence2vec(sentence)
        sims = self.index[sentence_vec]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_15 = sim_sort[0:k]

        return top_15

    def get_word_vector(self, s1,s2):
        """
        :param s1: 句子1
        :param s2: 句子2
        :return: 返回句子的余弦相似度
        """
        # 分词
        cut1 = jieba.cut(s1)
        cut2 = jieba.cut(s2)
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

    def similarity(self, sentence, k = 15):
        list_list_kanswer = self.get_topk_answer(sentence, k)

        x_sim = np.zeros((k, k))
        for i in range(k):
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
        return n_result, self.data_chat.iat[list_list_kanswer[n_result][0]+1, 6]

    def predict(self, session_list, session_length, session_text):
        with open("../output/" + "ans " + self.filename, "w") as f_out:
            cnt = 0
            for i in range(len(session_list)):
                f_out.write("<session " + session_list[i] + ">\n")
                for j in range(session_length[i]):
                    f_out.write(self.similarity(session_text[cnt])[1] + "\n")
                    cnt += 1
                f_out.write("</session " + session_list[i] + ">\n\n")



model_tfidf = Tfidf("chat_10per.txt")
model_tfidf.fit()
# n_result, sentence_result = model_tfidf.similarity("你好，请问增值税专用发票可以开吧")
# print(sentence_result)


data_loader = DataLoader()
session_list, session_length, session_text = data_loader.read_file()
model_tfidf.predict(session_list, session_length, session_text)
print("Finish")