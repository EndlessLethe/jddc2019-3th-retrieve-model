import time
from gensim import similarities
from code.embedding_model_loader import EmbeddingModelLoader
from code.unsupervised_reranker import UnsupervisedReranker
import pandas as pd
from code.jieba_seg import JiebaSeg
import numpy as np
import os
import pickle
from code.session_processer import SessionProcesser
from code.bert.run_classifier import run_classifier
import logging

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
        self.dict_q_index_to_data_index = None


    def load_data(self):
        data_chat = pd.read_csv(self.filepath_input, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
        # return data_chat[[0]]
        return data_chat[[6]]

    def fit(self, num_topics = None):
        self.data = self.load_data()
        input_name = self.filepath_input.split("/")[-1].replace(".txt", "")

        ## data sent to EmbeddingModelLoader is only questions or all data.
        # self.dict_q_index_to_data_index = None
        # data_q = self.data

        list_is_picked = self.get_dict_q_index_to_data_index()
        data_q = self.data[list_is_picked]

        ## load and train model
        self.model_loader = EmbeddingModelLoader(self.model_index, data_q, input_name, True, num_topics)


    ## 先使用sentence2vec将需要匹配的句子传进去
    def list_sentence_to_corpus(self, list_sentence):
        if self.model_index == 6:
            return self.model_loader.list_sentence_to_corpus_bert(list_sentence)

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


    def get_dict_q_index_to_data_index(self):
        dict_q2data = {}
        list_is_picked = []

        data_chat = pd.read_csv(self.filepath_input, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)[[2]]
        cnt = 0
        for i in range(data_chat.shape[0]):
            if data_chat.iat[i, 0] == 0:
                dict_q2data[cnt] = i
                cnt += 1
                list_is_picked.append(True)
            else :
                list_is_picked.append(False)
        self.dict_q_index_to_data_index = dict_q2data

        logging.info("Input QA Data Size: " + str(len(list_is_picked)))
        logging.info("Input Question Size: " + str(cnt))

        return list_is_picked

    def get_data_index(self, q_index):
        if self.dict_q_index_to_data_index != None:
            return self.dict_q_index_to_data_index[q_index]
        else :
            return q_index

    def get_list_q_candidate_index(self, texts, k):
        """
        list_q_candidate_index has a shape: n_q * k * 2
            The last dim is (sentence index, similarity score)
        """
        list_q_candidate_index = []

        corpus_vec = self.list_sentence_to_corpus(texts)
        for vec_sentence in corpus_vec:
            list_candidate = self.get_topk_answer(vec_sentence, k)
            list_q_candidate_index.append(list_candidate)

        list_q_candidate_index = self.reformat_list_q_candidate_index(list_q_candidate_index)
        return list_q_candidate_index

    def get_list_a_candidate_index(self, list_q_candiadate_index):
        list_a_candidate_index = []
        for list_k in list_q_candiadate_index:
            list_a_index = []
            for i in range(len(list_k)):
                list_a_index.append(list_k[i]+1)
            list_a_candidate_index.append(list_a_index)
        return list_a_candidate_index

    def get_topk_answer(self, vec_sentence, k):
        """求最相似的句子"""
        sims = self.model_loader.index[vec_sentence]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_k = sim_sort[0:k]

        return top_k

    def predict(self, filepath_input, filepath_result, k):
        session_list_id, session_length, session_list_q = SessionProcesser.read_file(filepath_input, use_context=False)

        ## get list_q_candidate and output to examine
        list_q_candidate_index = self.get_list_q_candidate_index(session_list_q, k)
        list_a_candidate_index = self.get_list_a_candidate_index(list_q_candidate_index)

        ## output candidate data for bert amd debugging
        filepath_q_candidate = "./out/q_candidate.txt"
        self.output_candidate(list_q_candidate_index, session_list_q, filepath_q_candidate)
        filepath_q_candidate = "./out/a_candidate.txt"
        self.output_candidate(list_a_candidate_index, session_list_q, filepath_q_candidate)
        filepath_bert_test = "./code/bert/data/test.tsv"
        self.output_candidate(list_a_candidate_index, session_list_q, filepath_bert_test)

        ## use unsupervised reranker
        list_q_index = self.get_unsupervised_reranker_result(list_q_candidate_index, k)
        logging.debug(str(list_q_index))
        list_answer = self.get_answer(list_q_index)
        list_question = self.get_sentence(list_q_index)
        SessionProcesser.output_file("./out/ur_answer.txt", session_list_id, session_length, session_list_q, list_answer)
        SessionProcesser.output_file("./out/ur_qusetion.txt", session_list_id, session_length, session_list_q, list_question)
        # SessionProcesser.output_file(filepath_result, session_list_id, session_length, session_list_q, list_answer)

        ## use bert as reranker
        run_classifier()
        list_q_index = self.get_bert_q_index(list_q_candidate_index, k)
        list_answer = self.get_answer(list_q_index)
        list_question = self.get_sentence(list_q_index)
        SessionProcesser.output_file("./out/bert_answer.txt", session_list_id, session_length, session_list_q, list_answer)
        SessionProcesser.output_file("./out/bert_qusetion.txt", session_list_id, session_length, session_list_q, list_question)
        SessionProcesser.output_file(filepath_result, session_list_id, session_length, session_list_q, list_answer)

        return list_q_index, list_answer

    def get_answer(self, list_q_index):
        list_answer = []
        for i in range(len(list_q_index)):
            list_answer.append(self.data.iat[list_q_index[i]+1, 0])
        return list_answer

    def get_sentence(self, list_sentence_index):
        list_sentence = []
        for i in range(len(list_sentence_index)):
            list_sentence.append(self.data.iat[list_sentence_index[i], 0])
        return list_sentence

    def get_unsupervised_reranker_result(self, list_q_candidate_index, k):
        ur = UnsupervisedReranker()
        list_q_index = []

        cnt = 0
        for list_candidate in list_q_candidate_index:
            q_rank = ur.similarity(list_candidate, self.data, k)
            list_q_index.append(list_q_candidate_index[cnt][q_rank])
            cnt += 1
        return list_q_index

    def reformat_list_q_candidate_index(self, list_q_candidate_index):
        """
        The input list_q_candidate_index has a quite strange shape : n_q * k * 2.
        And its index is data_q index not data_all index
        So this function is uesed to :
        1. reshape it as 2-d array with shape n_q * k.
        2. transform data_q index into data_all index
        """
        list_new = []
        for list_k in list_q_candidate_index:
            list_k_index = []
            for i in range(len(list_k)):
                list_k_index.append(self.get_data_index(list_k[i][0]))
            list_new.append(list_k_index)
        return list_new

    def output_candidate(self, list_q_candidate, session_list_q, filepath):
        with open(filepath, "w", encoding="utf-8") as f_out:
            for i in range(len(session_list_q)):
                for j in range(len(list_q_candidate[i])):
                    q_index = list_q_candidate[i][j]
                    f_out.write(session_list_q[i] + "\t" + self.data.iat[q_index, 0] + "\t0\n")
        print("output candidate file as bert format to:", filepath)

    def get_bert_q_index(self, list_q_candidate_index, k):
        x_lable = self.get_bert_result(k)
        list_q_index = []
        for i in range(x_lable.shape[0]):
            list_q_index.append(list_q_candidate_index[i][x_lable[i]])

        return list_q_index

    @classmethod
    def get_bert_result(cls, k):
        filepath_bert_result = "./bert/out/test_results.tsv"

        data_result = pd.read_csv(filepath_bert_result, header = None, sep = "\t")
        x_result = np.array(data_result[1])
        x_result = x_result.reshape((int(data_result.shape[0] / k), int(k)))

        x_lable = np.argmax(x_result, axis=1)
        return x_lable

# print(RunModel.get_bert_result(30))