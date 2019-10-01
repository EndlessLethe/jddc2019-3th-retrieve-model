#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from session_processer import SessionProcesser
from run_model import RunModel
from result_evaluator import ResultEvaluator
import logging

def main(filepath_quz, filepath_result):
    filepath_train = "data/chat_0.01per.txt"
    # filepath_train = "data/chat_1per with context.txt"
    rm = RunModel(filepath_train, 5)
    # rm.fit(num_topics = 80)
    rm.fit()
    rm.predict(filepath_quz, filepath_result, k = 30)

    filepath_test = "data/dev_answer.txt"
    re = ResultEvaluator(filepath_result, filepath_test)
    print(re.eval_result())


filepath_quz = "data/JDDC_评测用数据集/dev_question.txt"
filepath_result = "out/test.txt"
main(filepath_quz, filepath_result)

# main(sys.argv[1], sys.argv[2])


