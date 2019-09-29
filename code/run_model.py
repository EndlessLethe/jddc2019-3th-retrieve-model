import time
from gensim import similarities
from embedding_model_loader import EmbeddingModelLoader
from unsupervised_reranker import UnsupervisedReranker
import pandas as pd
from jieba_seg import JiebaSeg
import numpy as np
import os
import pickle


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
        filepath_index = "out/" + input_name + " " + str(self.model_index) + ".index"
        filepath_model = "out/" + input_name + " " + str(self.model_index) + ".model"
        filepath_dict = "out/" + input_name + " " + str(self.model_index) + ".dict.pkl"

        ## load and train model
        self.model_loader = EmbeddingModelLoader(self.model_index, self.data, filepath_index, filepath_model, filepath_dict, True, num_topics)


    ## 先使用sentence2vec将需要匹配的句子传进去
    def sentence2vec(self, sentence):
        # list_word = list(self.seg_jieba.cut(sentence, True))
        list_word = list(self.seg_jieba.cut(sentence, False))

        if self.model_index == 0:
            ## bow
            return self.model_loader.dictionary.doc2bow(list_word)
        elif self.model_index == 1:
            ## tfidf
            return self.model_loader.model[self.model_loader.dictionary.doc2bow(list_word)]
        elif self.model_index == 3 or self.model_index == 4:
            vec_tfidf = self.model_loader.tfidf_model[self.model_loader.dictionary.doc2bow(list_word)]
            return self.model_loader.model[vec_tfidf]
        elif self.model_index == 5:
            return EmbeddingModelLoader.text2corpus_elmo(self.model_loader.model, [list_word])[0]

    def get_topk_answer(self, sentence, k=15):
        """求最相似的句子"""
        vec_sentence = self.sentence2vec(sentence)
        sims = self.model_loader.index[vec_sentence]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_k = sim_sort[0:k]

        return top_k

    def predict(self, session_list, session_length, session_text, filepath_result, k = 15):
        ur = UnsupervisedReranker()

        with open(filepath_result, "w", encoding='utf-8') as f_out:
            cnt = 0
            for i in range(len(session_list)):
                f_out.write("<session " + session_list[i] + ">\n")
                for j in range(session_length[i]):
                    list_list_kanswer = self.get_topk_answer(session_text[cnt], k)
                    f_out.write(ur.similarity(list_list_kanswer, self.data, k)[1] + "\n")
                    cnt += 1
                f_out.write("</session " + session_list[i] + ">\n\n")

