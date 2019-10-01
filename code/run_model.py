import time
from gensim import similarities
from embedding_model_loader import EmbeddingModelLoader
from unsupervised_reranker import UnsupervisedReranker
import pandas as pd
from jieba_seg import JiebaSeg
import numpy as np
import os
import pickle
from session_processer import SessionProcesser


class RunModel():
    """
    This Class is used to connect all components, such as embedding part for training,
    and search part, rerank part for predicting.
    """
    def __init__(self, filepath_input, model_index):
        """
        Args:
            filepath_input: the filepath of training data
            model_index: use model_index to create the given embedding model.
            0 - bow, 1 - tfidf, 2 - bm25, 3 - lsi, 4 - lda, 5 - elmo
        """
        self.filepath_input = filepath_input
        self.data = None
        self.model_index = model_index
        self.seg_jieba = JiebaSeg()
        self.model_loader = None

    def load_data(self):
        data_chat = pd.read_csv(self.filepath_input, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
        # return data_chat[[0]]
        return data_chat[[6]]

    def fit(self, num_topics = None):
        self.data = self.load_data()

        input_name = self.filepath_input.split("/")[-1].replace(".txt", "")


        ## load and train model
        self.model_loader = EmbeddingModelLoader(self.model_index, self.data, input_name, True, num_topics)


    ## 先使用sentence2vec将需要匹配的句子传进去
    def texts2corpus(self, list_sentence):
        list_sentece_cuted = []
        for sentence in list_sentence:
            list_sentece_cuted.append(list(self.seg_jieba.cut(sentence, False)))

        if self.model_index == 0:
            return self.model_loader.text2corpus_bow(list_sentece_cuted)
        elif self.model_index == 1:
            return self.model_loader.text2corpus_tfidf(list_sentece_cuted)
        elif self.model_index == 3:
            return self.model_loader.text2corpus_lsi(list_sentece_cuted)
        elif self.model_index == 4:
            return self.model_loader.text2corpus_lda(list_sentece_cuted)
        elif self.model_index == 5:
            return self.model_loader.text2corpus_elmo(list_sentece_cuted)

        # corpus_vec = []
        # for sentence in list_sentence:
        #     sentence = list(self.seg_jieba.cut(sentence, False))
        #     if self.model_index == 0:
        #         ## bow
        #         corpus_vec.append(self.model_loader.dictionary.doc2bow(sentence))
        #     elif self.model_index == 1:
        #         ## tfidf
        #
        #         corpus_vec.append(self.model_loader.model[self.model_loader.dictionary.doc2bow(sentence)])
        #     elif self.model_index == 3 or self.model_index == 4:
        #         vec_tfidf = self.model_loader.tfidf_model[self.model_loader.dictionary.doc2bow(sentence)]
        #         corpus_vec.append(self.model_loader.model[vec_tfidf])
        # return corpus_vec

    def get_list_result(self, texts, k):
        list_q_index = []
        list_a_sentence = []
        ur = UnsupervisedReranker()
        corpus_vec = self.texts2corpus(texts)
        for vec_sentence in corpus_vec:
            list_candidate = self.get_topk_answer(vec_sentence, k)
            q_index, a_sentence = ur.similarity(list_candidate, self.data, k)
            list_q_index.append(q_index)
            list_a_sentence.append(a_sentence)
        return list_q_index, list_a_sentence

    def get_topk_answer(self, vec_sentence, k=15):
        """求最相似的句子"""
        sims = self.model_loader.index[vec_sentence]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_k = sim_sort[0:k]

        return top_k

    def predict(self, filepath_input, filepath_result, k = 15):
        session_list_id, session_length, session_list_q = SessionProcesser.read_file(filepath_input, use_context=False)
        list_q_index, list_a_sentence = self.get_list_result(session_list_q, k)
        SessionProcesser.output_file(filepath_result, session_list_id, session_length, session_list_q, list_a_sentence)
        return list_q_index





