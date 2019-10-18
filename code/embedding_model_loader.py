# coding=utf-8
import pandas as pd
import os
import numpy as np
import re
# import sklearn
from code.jieba_seg import JiebaSeg
import time
import logging
from gensim import corpora, models, summarization
import pickle
from gensim import similarities
import chardet

class EmbeddingModelLoader():
    """
    This class is a embedding model factory.
    """

    def __init__(self, model_index, data, input_name, is_load = True, num_topics = None):
        self.model_index = model_index
        self.dictionary = None
        self.tfidf_model = None
        self.model = None
        self.index = None
        self.seg_jieba = JiebaSeg()
        self.data_bert_embedding = None
        self.dict_word2index = None

        filepath_index =  "out/" + str(self.model_index) + " " + input_name + ".index"
        filepath_model = "out/" + str(self.model_index) + " " + input_name + ".model"
        filepath_tfidf_model = "out/" + str(self.model_index) + " " + input_name + ".tfidf_model"
        filepath_dict = "out/" + str(self.model_index) + " " + input_name + ".dict.pkl"
        filepath_dict_word2index = "out/" + str(self.model_index) + " " + input_name + ".dict_word2index.pkl"

        ## load model from existing files.
        if filepath_index != None and os.path.exists(filepath_index) and is_load == True:
            self.load_model(filepath_index, filepath_model, filepath_dict, filepath_tfidf_model, filepath_dict_word2index)

        ## when init embedding model, train it at once
        # logging.debug("Training model.")
        else:
            if self.model_index == 1:
                corpus_embedding = self.tfidf_fit(data)
            elif self.model_index == 3:
                corpus_embedding = self.lsi_fit(data, num_topics)
            elif self.model_index == 5:
                corpus_embedding = self.elmo_fit_large(data)
            elif self.model_index == 6:
                corpus_embedding = self.bert_fit(data)

            # logging.debug("Creating index.")
            self.index = self.get_index(filepath_index, corpus_embedding, num_topics)
            self.save_data(filepath_index, filepath_model, filepath_dict, filepath_tfidf_model, filepath_dict_word2index)

    def load_model(self, filepath_index, filepath_model, filepath_dict, filepath_tfidf_model = None, filepath_dict_word2index = None):
        # logging.debug("Loading existing model.")
        self.index = similarities.Similarity.load(filepath_index)
        if self.model_index == 6:
            self.use_bert_embedding()
            return

        with open(filepath_dict, 'rb') as f:
            self.dictionary = pickle.load(f)
        if self.model_index == 1:
            self.model = models.TfidfModel.load(filepath_model)
        elif self.model_index == 2:
            pass
        elif self.model_index == 3:
            self.model = models.LsiModel.load(filepath_model)
        elif self.model_index == 4:
            self.model = models.LdaModel.load(filepath_model)
        elif self.model_index == 5:
            from code.ELMo.elmoformanylangs.elmo import Embedder
            self.model = Embedder("./code/ELMo/zhs.model/", 16)

        if os.path.exists(filepath_tfidf_model):
            self.tfidf_model = models.TfidfModel.load(filepath_model)

    def save_data(self, filepath_index, filepath_model, filepath_dict, filepath_tfidf_model = None, filepath_dict_word2index = None):
        ## save the result
        self.index.save(filepath_index)

        if self.model_index == 6:
            return

        with open(filepath_dict, 'wb') as f:
            pickle.dump(self.dictionary, f)
        if self.model_index != 5:
            self.model.save(filepath_model)
            if self.tfidf_model != None:
                self.tfidf_model.save(filepath_tfidf_model)


    def bert_fit(self, data):
        self.use_bert_embedding()
        corpus_bert = self.data2corpus_bert(data)
        return corpus_bert

    def data2corpus_bert(self, data):
        data.columns = [0]
        corpus_bert = self.list_sentence_to_corpus_bert(data[0].values)
        return corpus_bert

    def list_sentence_to_corpus_bert(self, list_sentence):
        logging.debug("dict_word2index is initialed properly: " + str(len(self.dict_word2index)))

        corpus_bert = []
        cnt_empty = 0
        cnt_not_found = 0

        for i in range(len(list_sentence)):
            list_word_embedding = []
            sentence = list_sentence[i]

            seg_jieba = JiebaSeg()
            sentence_cuted = seg_jieba.cut(sentence)

            for j in range(len(sentence_cuted)):
                char_str = sentence[j]
                try:
                    index = self.dict_word2index[char_str]

                except KeyError:
                    cnt_not_found += 1
                    # logging.debug("char '" + char_str + "' is not in dict.")
                    continue

                word_embedding = list(self.data_bert_embedding.iloc[index])
                list_word_embedding.append(word_embedding)

            if len(list_word_embedding) != 0:
                corpus_sentence = []
                vec_sentence = np.sum(list_word_embedding, axis=0) / len(list_word_embedding)
                cnt = 0
                for vec in vec_sentence:
                    corpus_sentence.append((cnt, vec))
                    cnt += 1
            else:
                cnt_empty += 1
                corpus_sentence = []
            corpus_bert.append(corpus_sentence)

            if i % 1000 == 0 and i != 0:
                logging.info("Finished {0} sentences.".format(i))

        logging.debug("corpus_bert size: " + str(len(corpus_bert)))
        logging.debug("the first element in corpus_bert: " + str(corpus_bert[0]))
        logging.debug("cnt_empty in corpus_bert: " + str(cnt_empty))
        logging.debug("cnt_not_found in corpus_bert: " + str(cnt_not_found))

        return corpus_bert

    def use_bert_embedding(self):
        logging.info("Loading bert embedding. Wait a moment.")
        filepath_bert_data = "./code/bert/JDAI-WORD-EMBEDDING/JDAI-Word-Embedding.txt"
        data_bert = pd.read_csv(filepath_bert_data, sep = " ", skiprows = 1, header = None)
        data_bert.pop(301)
        data_word = data_bert.pop(0)
        logging.debug("bert embedding shape: " + str(data_bert.shape))
        dict_word2index = {}
        for i in range(data_word.shape[0]):
            dict_word2index[data_word[i]] = i
        self.dict_word2index = dict_word2index
        self.data_bert_embedding = data_bert

    def get_index(self, filepath_index, corpus_embedding, num_topics = None):
        filepath_index += ".tmp"

        if self.model_index == 0:
            ## bow
            index = similarities.Similarity(filepath_index, corpus_embedding, len(self.dictionary))
        elif self.model_index == 1:
            ## tfidf
            index = similarities.Similarity(filepath_index, corpus_embedding, len(self.dictionary))
        elif self.model_index == 3 or self.model_index == 4:
            index = similarities.Similarity(filepath_index, corpus_embedding, num_topics)
        elif self.model_index == 5:
            index = similarities.Similarity(filepath_index, corpus_embedding, 1024)
        elif self.model_index == 6:
            index = similarities.Similarity(filepath_index, corpus_embedding, 300)
        return index

    def get_texts(self, data):
        texts = []
        for i in range(data.shape[0]):
            sentence = data.iat[i, 0]
            list_word = list(self.seg_jieba.cut(sentence, False))
            texts.append(list_word)
        return texts

    def bow_fit(self, data):
        texts = self.get_texts(data)
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=100000, keep_tokens=None)

        corpus_bow = self.text2corpus_bow(texts)
        return corpus_bow

    def tfidf_fit(self, data):
        corpus_bow = self.bow_fit(data)
        self.model = models.TfidfModel(corpus_bow)
        self.tfidf_model = self.model
        corpus_tfidf = self.corpus_bow2tfidf(corpus_bow)

        return corpus_tfidf

    def bm25_fit(self, data):
        corpus_bow = self.bow_fit(data)
        self.model = summarization.bm25.BM25(corpus_bow)
        corpus_bm25 = self.corpus_bow2bm25(corpus_bow)
        return corpus_bm25

    def lsi_fit(self, data, num_topics):
        corpus_tfidf = self.tfidf_fit(data)
        self.model = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=num_topics)
        corpus_lsi = self.corpus_tfidf2lsi(corpus_tfidf)
        return self.dictionary, self.model,  corpus_lsi, self.tfidf_model

    def lda_fit(self, data, num_topics):
        corpus_tfidf = self.tfidf_fit(data)
        self.model = models.LdaModel(corpus_tfidf, id2word=self.dictionary, num_topics=num_topics)
        corpus_lda = self.corpus_tfidf2lda(corpus_tfidf)
        return corpus_lda

    def elmo_fit_small(self, data):
        from code.ELMo.elmoformanylangs.elmo import Embedder
        self.model = Embedder("./code/ELMo/zhs.model/", 64)
        texts = self.get_texts(data)
        self.dictionary = corpora.Dictionary(texts)

        corpus_elmo = self.text2corpus_elmo(texts)
        return corpus_elmo

    def elmo_fit_large(self, data):
        from code.ELMo.elmoformanylangs.elmo import Embedder
        self.model = Embedder("./code/ELMo/zhs.model/", 16)
        texts = self.get_texts(data)
        self.dictionary = corpora.Dictionary(texts)

        corpus_elmo = []
        list_sentence = []
        for i in range(data.shape[0]):
            if i % 10000 == 0 and i != 0:
                list_sentence.append(texts[i])
                list_word_embedding = self.model.sents2elmo(list_sentence)
                for j in range(len(list_word_embedding)):
                    corpus_sentence = []
                    vec_sentence = np.sum(list_word_embedding[j], axis = 0) / len(list_word_embedding[j])
                    cnt = 0
                    for vec in vec_sentence:
                        corpus_sentence.append((cnt, vec))
                        cnt += 1
                    corpus_elmo.append(corpus_sentence)
                list_sentence = []

                logging.info('Finished {0} sentences.'.format(i))
            else :
                list_sentence.append(texts[i])

        if len(list_sentence) != 0:
            list_word_embedding = self.model.sents2elmo(list_sentence)
            for j in range(len(list_word_embedding)):
                corpus_sentence = []
                vec_sentence = np.sum(list_word_embedding[j], axis=0) / len(list_word_embedding[j])
                cnt = 0
                for vec in vec_sentence:
                    corpus_sentence.append((cnt, vec))
                    cnt += 1
                corpus_elmo.append(corpus_sentence)


        return corpus_elmo


    def text2corpus_elmo(self, texts):
        list_word_embedding = self.model.sents2elmo(texts)
        list_sentence_vec = []
        for sentence in list_word_embedding:
            list_sentence_vec.append(np.sum(sentence, axis=0) / len(sentence))

        corpus_text = []
        for list_vec in list_sentence_vec:
            corpus_sentence = []
            cnt = 0
            for vec in list_vec:
                corpus_sentence.append((cnt, vec))
                cnt += 1
            corpus_text.append(corpus_sentence)
        return corpus_text

    def text2corpus_bow(self, texts):
        corpus_bow = [self.dictionary.doc2bow(text) for text in texts]
        return corpus_bow

    def corpus_bow2tfidf(self, corpus_bow):
        return self.tfidf_model[corpus_bow]

    def text2corpus_tfidf(self, texts):
        corpus_bow = self.text2corpus_bow(texts)
        corpus_tfidf = self.tfidf_model[corpus_bow]
        return corpus_tfidf

    def corpus_bow2bm25(self, corpus_bow):
        corpus_bm25 = self.model[corpus_bow]
        return corpus_bm25

    def corpus_tfidf2lsi(self, corpus):
        corpus_lsi = self.model[corpus]
        return corpus_lsi

    def corpus_tfidf2lda(self, corpus):
        corpus_lda = self.model[corpus]
        return corpus_lda



# filepath_quz = "../data/JDDC_100W训练数据集/训练数据集/chat_1per.txt"
# filepath_result = "../output/ans.txt"
# model_tfidf = Tfidf(filepath_quz, filepath_result)
# model_tfidf.fit()
#
