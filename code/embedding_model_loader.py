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
from gensim import corpora, models, summarization
from session_loader import SessionLoader


class EmbeddingModelLoader():
    def __init__(self):
        self.dictionary = None
        self.model = None
        self.index = None
        self.seg_jieba = JiebaSeg()

    def tfidf_fit(self, data):
        corpus_bow = self.get_corpus_bow(data)
        corpus_tfidf = self.get_corpus_tfidf(corpus_bow)
        return self.dictionary, self.model, corpus_tfidf

    def get_corpus_bow(self, data):
        texts = []

        for i in range(data.shape[0]):
            sentence = data.iat[i, 0]
            # list_word = list(self.seg_jieba.cut(sentence, True))
            list_word = list(self.seg_jieba.cut(sentence, False))
            texts.append(list_word)
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=100000, keep_tokens=None)
        corpus_bow = [self.dictionary.doc2bow(text) for text in texts]
        return corpus_bow

    def get_corpus_tfidf(self, corpus_bow):
        self.model = models.TfidfModel(corpus_bow)
        corpus_tfidf = self.model[corpus_bow]
        return corpus_tfidf

    def bm25_fit(self, data):
        corpus_bow = self.get_corpus_bow(data)
        corpus_bm25 = self.get_corpus_bm25(corpus_bow)
        return self.model, corpus_bm25

    def get_corpus_bm25(self, corpus_bow):
        self.model = summarization.bm25.BM25(corpus_bow)
        corpus_bm25 = self.model[corpus_bow]
        return corpus_bm25

    def lsi_fit(self, data, num_topics):
        corpus_bow = self.get_corpus_bow(data)
        corpus_tfidf = self.get_corpus_tfidf(corpus_bow)
        corpus_lsi = self.get_corpus_lsi(corpus_tfidf, num_topics)
        return self.dictionary, self.model, corpus_lsi

    def get_corpus_lsi(self, corpus, num_topics):
        """
        corpus can be corpus_tfidf or corpus_bow
        """
        self.model = models.LsiModel(corpus, id2word=self.dictionary, num_topics=num_topics)
        corpus_lsi = self.model[corpus]
        return corpus_lsi

    def lda_fit(self, data, num_topics):
        corpus_bow = self.get_corpus_bow(data)
        corpus_tfidf = self.get_corpus_tfidf(corpus_bow)
        corpus_lda = self.get_corpus_lda(corpus_tfidf, num_topics)
        return self.dictionary, self.model, corpus_lda

    def get_corpus_lda(self, corpus, num_topics):
        """
        corpus can be corpus_tfidf or corpus_bow
        """
        self.model = models.LdaModel(corpus, id2word=self.dictionary, num_topics=num_topics)
        corpus_lda = self.model[corpus]
        return corpus_lda



# filepath_origin = "../data/JDDC_100W训练数据集/训练数据集/chat_1per.txt"
# filepath_result = "../output/ans.txt"
# model_tfidf = Tfidf(filepath_origin, filepath_result)
# model_tfidf.fit()
#
