#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from code.data_loader import DataLoader
from code.tfidf import Tfidf

def main(filepath_input, filepath_result):
    filepath_train = "data/chat_10per.txt"
    model_tfidf = Tfidf(filepath_train)
    model_tfidf.fit()

    data_loader = DataLoader(filepath_input)
    session_list, session_length, session_text = data_loader.read_file()
    model_tfidf.predict(session_list, session_length, session_text, filepath_result)

# filepath_input = "data/JDDC_评测用数据集/dev_question.txt"
# filepath_result = "output/test.txt"
# main(filepath_input, filepath_result)

main(sys.argv[1], sys.argv[2])
