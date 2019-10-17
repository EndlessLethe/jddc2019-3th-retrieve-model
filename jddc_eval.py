#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from code.run_model import RunModel
from code.result_evaluator import ResultEvaluator
import logging

l_g = logging.getLogger()
l_g.setLevel(logging.DEBUG)

def main(filepath_quz, filepath_result):
    filepath_train = "data/chat_10per.txt"
    filepath_answer = "data/dev_answer.txt"

    rm = RunModel(filepath_train, 1)

    rm.fit()
    rm.predict(filepath_quz, filepath_result, k = 30)

    re = ResultEvaluator("./out/ur_answer.txt", filepath_answer)
    print(re.eval_result())

    re = ResultEvaluator(filepath_result, filepath_answer)
    print(re.eval_result())

def human_eval():
    l_g.setLevel(logging.WARNING)
    print('=' * 50 + '请稍候...' + '=' * 50)
    filepath_train = "data/chat_1per.txt"
    rm = RunModel(filepath_train, 5)
    rm.fit()
    print('=' * 48 + '模型加载完成' + '=' * 48)
    while True:
        question = input('用户Input：')
        if question == 'end':
            break
        elif question == 'finish':
            print('change session')
        else:
            answer = rm.predict_single_task(question=question, k=30)
            print(answer)

filepath_quz = "data/dev_question.txt"
filepath_result = "out/test.txt"
main(filepath_quz, filepath_result)

# human_eval()
# main(sys.argv[1], sys.argv[2])


