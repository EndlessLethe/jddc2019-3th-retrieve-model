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
import logging
from code.rule_based_processor import process_question_list

class RunModel():
    """
    This Class is used to connect all components, such as embedding part for training,
    and retrieve part, re-rank part for predicting.
    """
    def __init__(self, filepath_train, model_index, use_bert = True, use_history = True):

        """
        Args:
            filepath_train: the filepath of training data
            model_index: use model_index to create the given embedding model.
            0 - bow, 1 - tfidf, 2 - bm25, 3 - lsi, 4 - lda, 5 - elmo
        """
        self.filepath_train = filepath_train

        self.model_index = model_index
        self.seg_jieba = JiebaSeg()
        self.model_loader = None
        self.dict_q_index_to_data_index = None
        self.data = self.load_data()
        self.use_bert = use_bert
        self.use_history = use_history

        ## data sent to EmbeddingModelLoader is only questions.
        self.list_is_picked = self.get_dict_q_index_to_data_index()


    def load_data(self):
        data_chat = pd.read_csv(self.filepath_train, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
        # return data_chat[[0]]
        return data_chat[[6]]

    def fit(self, num_topics = None):
        ## filepath is like : "data/chat_0.1per.txt" and then get "chat_0.1per"
        input_name = self.filepath_train.split("/")[-1].replace(".txt", "")
        data_q = self.data[self.list_is_picked]

        ## load and train model
        self.model_loader = EmbeddingModelLoader(self.model_index, data_q, input_name, True, num_topics)

    ## 先使用sentence2vec将需要匹配的句子传进去
    def list_sentence_to_corpus(self, list_sentence):
        if self.model_index == 6:
            return self.model_loader.list_sentence_to_corpus_skipgram(list_sentence)

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

        data_chat = pd.read_csv(self.filepath_train, sep="\t", engine="python",
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
        return self.dict_q_index_to_data_index[q_index]


    def get_list_q_candidate_index(self, session_list_q, k):
        """
        list_q_candidate_index has a shape: n_total_q * k * 2
            The last dim is (sentence index, similarity score)
        So use reformat_list_q_candidate_index to reformat:
            shape becomes (n_total_q * k)
        """
        list_q_candidate_index = []
        texts = []
        for i in range(len(session_list_q)):
            for j in range(len(session_list_q[i])):
                texts.append(session_list_q[i][j])

        ## use model loaded to tranform text into corpus
        corpus_vec = self.list_sentence_to_corpus(texts)

        ## use sentence corpus to compute most simlilar k question
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
        """
        The shape of session_list_q is (n_session, n_q)
        The shape of session_list_history is (n_session, n_history)
        The shape of list_q_candidate_index (n_total_q, k)
        """
        session_list_id, session_length, session_list_q, session_list_history = SessionProcesser.read_file(filepath_input)

        logging.debug("Num of sessions: " + str(len(session_list_id)))
        # logging.debug("Num of questions: " + str(len(session_list_q)))
        # logging.debug("Num of histories: " + str(len(session_list_history)))
        # logging.debug("histories: " + str(session_list_history))

        ## get list_q_candidate and output to examine
        list_q_candidate_index = self.get_list_q_candidate_index(session_list_q, k)
        logging.debug("list_q_candidate_index: " + str(list_q_candidate_index))
        list_a_candidate_index = self.get_list_a_candidate_index(list_q_candidate_index)

        ## output candidate data for bert and debugging
        filepath_q_candidate = "./out/q_candidate.txt"
        self.output_candidate(list_q_candidate_index, session_list_q, filepath_q_candidate)
        filepath_q_candidate = "./out/a_candidate.txt"
        self.output_candidate(list_a_candidate_index, session_list_q, filepath_q_candidate)

        ## use unsupervised reranker
        list_q_index = self.get_unsupervised_reranker_result(list_q_candidate_index, k)
        logging.debug("list_q_index: " + str(list_q_index))
        list_answer = self.get_answer(list_q_index)
        list_question = self.get_sentence(list_q_index)
        self.output_result("ur" , session_list_id, session_length, session_list_q, list_answer, list_question, filepath_result)

        ## use bert as reranker
        if self.use_bert:
            list_answer, list_question = self.run_bert(list_a_candidate_index, session_list_q, session_list_history,
                          list_q_candidate_index, k)
            self.output_result("bert" , session_list_id, session_length, session_list_q, list_answer, list_question, filepath_result)


        # ## use task dialog to provide standard answer for matched questions
        # list_sessoin_id = []
        # for i in range(len(session_list_id)):
        #     for j in range(session_length[i]):
        #         list_sessoin_id.append(session_list_id[i])
        # list_flag, list_answer_task = process_question_list(list_sessoin_id, session_list_q)
        # logging.debug("Task Dialog Len: " + str(len(list_flag)) + " " + str(len(list_answer_task)))
        # list_answer_multi = []
        # for i in range(len(session_list_q)):
        #     if list_flag[i]:
        #         list_answer_multi.append(list_answer_task[i])
        #     else:
        #         list_answer_multi.append(list_answer[i])
        # SessionProcesser.output_result_file("./out/task_answer.txt", session_list_id, session_length, session_list_q, list_flag)
        # SessionProcesser.output_result_file(filepath_result, session_list_id, session_length, session_list_q, list_answer_multi)

        return list_q_index, list_answer

    def run_bert(self, list_a_candidate_index, session_list_q, session_list_history,
                 list_q_candidate_index, k):
        # filepath_bert_predict = "./code/bert/data/test.tsv"
        # if not self.use_history:
        #     logging.info("Bert classifier for single dialog is running.")
        #     self.output_candidate(list_a_candidate_index, session_list_q, filepath_bert_predict)
        #     from code.bert.run_classifier_single import run_classifier
        #     run_classifier()
        # else:
        #     logging.info("Bert classifier for multi dialog is running.")
        #     self.output_candidate_with_history(list_a_candidate_index, session_list_q, session_list_history,
        #                                        filepath_bert_predict)
        #     from code.bert.run_classifier_multi import run_classifier
        #     run_classifier()
        list_q_index = self.get_bert_q_index(list_q_candidate_index, k, self.use_history)
        list_answer = self.get_answer(list_q_index)
        list_question = self.get_sentence(list_q_index)
        return list_answer, list_question


    def output_result(self, model_name, session_list_id, session_length, session_list_q, list_answer, list_question, filepath_result):
        SessionProcesser.output_result_file("./out/" + model_name + "_answer.txt", session_list_id, session_length, session_list_q,
                                            list_answer)
        SessionProcesser.output_result_file("./out/" + model_name + "_qusetion.txt", session_list_id, session_length, session_list_q,
                                            list_question)
        SessionProcesser.output_result_file(filepath_result, session_list_id, session_length, session_list_q,
                                            list_answer)

    def predict_single_task(self, question, k, predict_input_fn, estimator):
        session_list_q = []
        session_list_q.append(question)
        list_flag, list_answer = process_question_list([0], session_list_q)

        if list_flag[0]:
            logging.debug("task dialog: " + str(list_flag[0]) + list_answer[0])
            return list_answer[0]

        list_q_candidate_index = self.get_list_q_candidate_index(session_list_q, k)
        list_a_candidate_index = self.get_list_a_candidate_index(list_q_candidate_index)

        filepath_q_candidate = "./out/q_candidate_single_task.txt"
        self.output_candidate(list_q_candidate_index, session_list_q, filepath_q_candidate)

        filepath_bert_predict = "./code/bert/data/test.tsv"
        self.output_candidate(list_a_candidate_index, session_list_q, filepath_bert_predict)
        import code

        list_q_index = self.get_unsupervised_reranker_result(list_q_candidate_index, k)
        list_answer = self.get_answer(list_q_index)
        # print("retrieved model with unsupervised reranker: " + list_answer[0])

        code.bert.run_classifier.predict(predict_input_fn, estimator)
        list_q_index = self.get_bert_q_index(list_q_candidate_index, k)
        list_answer = self.get_answer(list_q_index)
        # print("retrieved model with bert: " + list_answer[0])

        return list_answer[0]

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
        logging.info("Unsupervised Reranker is running.")

        ur = UnsupervisedReranker()
        list_q_index = []

        cnt = 0
        for list_candidate in list_q_candidate_index:
            q_rank = ur.similarity(list_candidate, self.data, k)
            list_q_index.append(list_q_candidate_index[cnt][q_rank])
            # print("ur result", q_rank, list_q_candidate_index[cnt][q_rank])
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
        """
        The shape of session_list_q is (n_session, n_q)
        The shape of session_list_history is (n_session, n_history)
        The shape of list_q_candidate_index (n_total_q, k)
        """
        with open(filepath, "w", encoding="utf-8") as f_out:
            cnt = 0
            for i in range(len(session_list_q)):
                for j in range(len(session_list_q[i])):
                    for k in range(len(list_q_candidate[i])): # equals to top k
                        q_index = list_q_candidate[cnt][k]
                        f_out.write(session_list_q[i][j] + "\t" + self.data.iat[q_index, 0] + "\t0\n")
                    cnt += 1
        logging.info("output candidate file as bert format to: " + filepath)

    def output_candidate_with_history(self, list_q_candidate, session_list_q, session_list_history, filepath):
        with open(filepath, "w", encoding="utf-8") as f_out:
            for i in range(len(session_list_q)):
                logging.debug(str(session_list_history[i]))
                if len(session_list_history[i]) < 6:
                    continue
                assert len(session_list_history[i]) == 6
                list_his = []
                list_his.append(session_list_history[i][0])
                list_his.append(session_list_history[i][2])
                list_his.append(session_list_history[i][4])
                list_his.extend(session_list_q[i])
                for k in range(len(session_list_q[i])):
                    for j in range(len(list_q_candidate[i])):
                        h1 = list_his[k+1]
                        h2 = list_his[k+2]
                        h3 = list_his[k+3]
                        # logging.debug("h3" + h3 + " " + session_list_q[i][k])
                        assert h3 == session_list_q[i][k]
                        q_index = list_q_candidate[i][j]
                        f_out.write(h1 + "\t" + h2 + "\t" + h3 + "\t" + self.data.iat[q_index, 0] + "\t0\n")
        logging.info("output candidate file as bert format to: " + filepath)

    def get_bert_q_index(self, list_q_candidate_index, k, use_history):
        x_lable = self.get_bert_result(k, use_history)
        list_q_index = []
        for i in range(x_lable.shape[0]):
            list_q_index.append(list_q_candidate_index[i][x_lable[i]])
        return list_q_index

    @classmethod
    def get_bert_result(cls, k, use_history):
        if use_history:
            filepath_bert_result = "./code/bert/out/test_results.tsv"
        else :
            filepath_bert_result = "./code/bert/out_single/test_results.tsv"

        data_result = pd.read_csv(filepath_bert_result, header = None, sep = "\t")
        x_result = np.array(data_result[1])
        x_result = x_result.reshape((int(data_result.shape[0] / k), int(k)))

        x_lable = np.argmax(x_result, axis=1)
        logging.debug("bert result: " + str(x_lable))
        return x_lable
