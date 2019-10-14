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
    filepath_train = "data/chat_0.1per.txt"
    filepath_answer = "data/dev_answer.txt"

    rm = RunModel(filepath_train, 5)
    # rm.fit(num_topics = 80)

    rm.fit()
    rm.predict(filepath_quz, filepath_result, k = 30)

    re = ResultEvaluator("./out/ur_result.txt", filepath_answer)
    print(re.eval_result())

    re = ResultEvaluator(filepath_result, filepath_answer)
    print(re.eval_result())


filepath_quz = "data/dev_question.txt"
filepath_result = "out/test.txt"
main(filepath_quz, filepath_result)

# main(sys.argv[1], sys.argv[2])


