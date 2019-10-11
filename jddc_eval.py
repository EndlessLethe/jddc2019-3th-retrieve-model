#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from code.run_model import RunModel
from code.result_evaluator import ResultEvaluator
import logging

def main(filepath_quz, filepath_result):
    filepath_train = "data/chat_1per.txt"
    filepath_test = "data/dev_answer.txt"

    rm = RunModel(filepath_train, 1)
    # rm.fit(num_topics = 80)

    rm.fit()
    rm.predict(filepath_quz, filepath_result, k = 30)

    re = ResultEvaluator(filepath_result, filepath_test)
    print(re.eval_result())


filepath_quz = "data/dev_question.txt"
filepath_result = "out/test.txt"
main(filepath_quz, filepath_result)

# main(sys.argv[1], sys.argv[2])


