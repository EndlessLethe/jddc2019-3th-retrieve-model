#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from session_loader import SessionLoader
from tfidf import Tfidf
from result_evaluator import ResultEvaluator

def main(filepath_input, filepath_result):
    filepath_train = "data/chat_1per.txt"
    # filepath_train = "data/chat_1per with context.txt"
    model_tfidf = Tfidf(filepath_train)
    model_tfidf.fit()

    data_loader = SessionLoader(filepath_input, use_context = False)
    session_list, session_length, session_text = data_loader.read_file()
    model_tfidf.predict(session_list, session_length, session_text, filepath_result, k = 10)

    filepath_test = "data/dev_answer.txt"
    re = ResultEvaluator(filepath_result, filepath_test)
    print(re.eval_result())


# filepath_origin = "data/JDDC_评测用数据集/dev_question.txt"
# filepath_result = "output/test.txt"
# main(filepath_origin, filepath_result)

main(sys.argv[1], sys.argv[2])


#char 1per 不去掉停用词