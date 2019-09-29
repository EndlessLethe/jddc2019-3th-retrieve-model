#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from session_loader import SessionLoader
from run_model import RunModel
from result_evaluator import ResultEvaluator
import logging

def main(filepath_input, filepath_result):
    filepath_train = "data/chat_0.01per.txt"
    # filepath_train = "data/chat_1per with context.txt"
    rm = RunModel(filepath_train, 5)
    # rm.fit(num_topics = 80)
    rm.fit()

    data_loader = SessionLoader(filepath_input, use_context = False)
    session_list, session_length, session_text = data_loader.read_file()
    rm.predict(session_list, session_length, session_text, filepath_result, k = 30)

    filepath_test = "data/dev_answer.txt"
    re = ResultEvaluator(filepath_result, filepath_test)
    print(re.eval_result())


filepath_origin = "data/JDDC_评测用数据集/dev_question.txt"
filepath_result = "out/test.txt"
main(filepath_origin, filepath_result)

# main(sys.argv[1], sys.argv[2])


