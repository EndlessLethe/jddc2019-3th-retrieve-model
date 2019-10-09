# coding=utf-8
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re
# import sklearn


class SessionProcesser():
    """
    这个类使用来处理测试集数据的。（训练集和开发集数据都是处理好的.csv文件，可以直接读取）
    测试集数据是以session形式给出的多个QQAQA对

    输出的数据是list_session_id, list_list_question
    """


    @classmethod
    def read_file(cls, file_path, use_context = True):
        with open(file_path, "r", encoding="utf8") as f:
            session_list_id = []
            session_length = []
            session_list_q = []
            line = f.readline()
            ques = []
            tlist = ""
            while line:
                string = line.strip()
                if string.startswith("<session"):
                    session_id = string[9:-1]
                    session_list_id.append(session_id)
                if string.startswith("</session"):
                    if session_list_id[-1]!=string[10:-1]:
                        raise ValueError(string, session_list_id[-1])
                    session_length.append(len(ques))
                    for q in ques:
                        session_list_q.append(tlist+""+q)
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
        return session_list_id, session_length, session_list_q

    @classmethod
    def output_file(cls, filepath_output, session_list_id, session_length, session_list_q, list_a_sentence):
        with open(filepath_output, "w", encoding='utf-8') as f_out:
            cnt = 0
            for i in range(len(session_list_id)):
                f_out.write("<session " + session_list_id[i] + ">\n")
                for j in range(session_length[i]):
                    f_out.write(list_a_sentence[cnt] + "\n")
                    cnt += 1
                f_out.write("</session " + session_list_id[i] + ">\n\n")
        print("Output result file in:", filepath_output)

# data_loader = SessionProcesser()
# session_list, session_length, session_text = data_loader.read_file()
# print(len(session_list))
# print(session_list)
# print(len(session_length))
# print(session_length)
# print(len(session_text))
# print(session_text[0])
# print(session_text[1])