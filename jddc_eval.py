#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from code.run_model import RunModel
from code.result_evaluator import ResultEvaluator
import logging
from code.data_processer import DataProcesser

l_g = logging.getLogger()
l_g.setLevel(logging.DEBUG)

def main(filepath_quz, filepath_result):
    filepath_train = "data/chat_1per.txt"
    filepath_answer = "data/dev_answer.txt"

    if not os.path.exists(filepath_train):
        dp = DataProcesser(7, 7)
        dp.get_file_primary([0, 1, 2, 3, 4], is_replace=False)

        ## adjust this function arg "k_per" to select k percentage data
        k = float(filepath_train.split("/")[-1].strip("chat_").strip("per.txt"))
        dp.get_file_middle(k)

        ## get bert train file
        # DataProcesser.generate_train_file_bert_single()

    ## __init__(self, filepath_train, model_index, use_bert = True, use_history = True)
    # rm = RunModel(filepath_train, 6, False, True)
    rm = RunModel(filepath_train, 6, True, True)
    # rm = RunModel(filepath_train, 6, True, False)

    rm.fit()
    rm.predict(filepath_quz, filepath_result, k = 30)

    re = ResultEvaluator("./out/ur_answer.txt", filepath_answer)
    print(re.eval_result())

    re = ResultEvaluator("./out/bert_answer.txt", filepath_answer)
    print(re.eval_result())

    re = ResultEvaluator(filepath_result, filepath_answer)
    print(re.eval_result())


## This function is unavailable because of the lack of other model part, as generate model or rule_based model.
def human_eval():
    print('=' * 50 + '请稍候...加载模型中' + '=' * 50)
    import tensorflow
    from code.bert.run_classifier import run_classifier

    tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

    l_g.setLevel(logging.DEBUG)


    filepath_train = "data/chat_10per.txt"
    rm = RunModel(filepath_train, 6)
    rm.fit()

    list_a_candidate_index = [[0]*30]

    session_list_q = [""]
    filepath_bert_predict = "./code/bert/data/test.tsv"
    rm.output_candidate(list_a_candidate_index, session_list_q, filepath_bert_predict)

    predict_input_fn, estimator = run_classifier()

    print('=' * 48 + '模型加载完成' + '=' * 48)

    l_g.setLevel(logging.ERROR)

    while True:
        question = input('用户Input：')
        if question == 'end':
            break
        elif question == 'finish':
            print('change session')
        else:
            answer = rm.predict_single_task(question, 30, predict_input_fn, estimator)
            print(answer)


filepath_quz = "data/dev_question.txt"
filepath_result = "out/test.txt"
main(filepath_quz, filepath_result)

# human_eval()
# main(sys.argv[1], sys.argv[2])


