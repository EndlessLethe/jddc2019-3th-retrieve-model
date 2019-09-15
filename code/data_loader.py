# coding=utf-8
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re
# import sklearn


class DataLoader():
    """
    这个类使用来处理测试集数据的。（训练集和开发集数据都是处理好的.csv文件，可以直接读取）
    测试集数据是以session形式给出的多个QQAQA对

    输出的数据是list_session_id, list_list_question
    """

    def __init__(self, file_path = "../data/JDDC_评测用数据集/dev_question.txt"):
        self.file_path = file_path

    def read_file(self, use_context = True):
        with open(self.file_path, "r", encoding="utf8") as f:
            newsess = ""
            session_list = []
            session_length = []
            session_text = []
            line = f.readline()
            ques = []
            tlist = ""
            while line:
                string = line.strip()
                if string.startswith("<session"):
                    session_id = string[9:-1]
                    session_list.append(session_id)
                if string.startswith("</session"):
                    if session_list[-1]!=string[10:-1]:
                        raise ValueError(string, session_list[-1])
                    session_length.append(len(ques))
                    for q in ques:
                        session_text.append(tlist+""+q)
                    ques = []
                    if use_context == False:
                        tlist = ""
                    else :
                        tlist += text.replace("!@@@!", " ") + " "
                    line = f.readline()
                if string == "<context>":
                    while True:
                        line = f.readline().strip()
                        if line.startswith("Q:") or line.startswith("A:"):
                            line = line[2:]
                            string = line.strip()
                            text = string.split("<sep>")[0]
                            # tlist += text.replace("!@@@!", "。") + "。"
                        else:
                            break
                if re.match(r"^<Q[0-9]*>(.*)</Q[0-9]*>$", string):
                    ques.append(re.match(r"^<Q[0-9]*>(.*)</Q[0-9]*>$", string)[1].replace("!@@@!", "。"))
                line = f.readline()
        return session_list, session_length, session_text

# data_loader = DataLoader()
# session_list, session_length, session_text = data_loader.read_file()
# print(len(session_list))
# print(session_list)
# print(len(session_length))
# print(session_length)
# print(len(session_text))
# print(session_text[0])
# print(session_text[1])