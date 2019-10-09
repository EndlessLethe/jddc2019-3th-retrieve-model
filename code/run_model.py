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
from code.bert.run_classifier import run_classifier

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
        self.k = None
        self.session_list_id = None
        self.session_length = None
        self.session_list_q = None
        self.list_q_index = None
        self.list_q_candidate_index = None
        self.list_a_sentence = None
        self.list_bert_sentence = None

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
        list_q_candidate_index = []
        list_a_sentence = []

        ur = UnsupervisedReranker()
        corpus_vec = self.texts2corpus(texts)
        for vec_sentence in corpus_vec:
            list_candidate = self.get_topk_answer(vec_sentence, k)

            q_index = ur.similarity(list_candidate, self.data, k)
            list_q_index.append(q_index)
            list_a_sentence.append(self.data.iat[q_index + 1, 0])

            list_q_candidate_index.append(list_candidate)
        return list_q_index, list_q_candidate_index, list_a_sentence

    def get_topk_answer(self, vec_sentence, k=15):
        """求最相似的句子"""
        sims = self.model_loader.index[vec_sentence]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_k = sim_sort[0:k]

        return top_k

    def predict(self, filepath_input, filepath_result, k = 15):
        self.session_list_id, self.session_length, self.session_list_q = SessionProcesser.read_file(filepath_input, use_context=False)
        self.k = k

        self.list_q_index, self.list_q_candidate_index, self.list_a_sentence = self.get_list_result(self.session_list_q, k)
        SessionProcesser.output_file(filepath_result, self.session_list_id, self.session_length, self.session_list_q, self.list_a_sentence)
        return self.list_q_index, self.list_q_candidate_index, self.list_a_sentence, self.session_list_q

    def run_bert(self):
        self.output_bert_format(self.list_q_candidate_index, self.session_list_q)
        run_classifier()
        self.find_bert_ans(self.list_q_candidate_index)

    def output_bert_format(self, list_q_candidate, session_list_q):
        filepath_bert_test = "./code/bert/JDAI-BERT/test.tsv"
        with open(filepath_bert_test, "w", encoding="utf-8") as f_bert:
            for i in range(len(session_list_q)):
                for j in range(len(list_q_candidate[i])):
                    q_index = list_q_candidate[i][j][0]
                    f_bert.write(session_list_q[i] + "\t" + self.data.iat[q_index, 0] + "\t0\n")
        print("output Bert QQ pair file.")

    def find_bert_ans(self, list_q_candidate_index):
        filepath_bert_result = "./code/bert/out/test_results.tsv"
        filepath_result = "./out/bert_reuslt.txt"

        data_result = pd.read_csv(filepath_bert_result, header = None, sep = "\t")
        x_result = np.array(data_result[0])
        x_result = x_result.reshape((int(data_result.shape[0] / self.k), int(self.k)))
        print(x_result.shape)
        x_lable = np.argmax(x_result, axis=1)
        print(x_lable.shape)
        print(x_lable)

        list_bert_sentence = []
        for i in range(x_lable.shape[0]):
            q_index = list_q_candidate_index[i][x_lable[i]][0]
            list_bert_sentence.append(self.data.iat[q_index+1, 0])

        self.list_bert_sentence = list_bert_sentence
        SessionProcesser.output_file(filepath_result, self.session_list_id, self.session_length, self.session_list_q, list_bert_sentence)
