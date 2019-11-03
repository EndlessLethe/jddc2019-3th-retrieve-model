import pandas as pd
import os
import re
import numpy as np
from code.data_analyzer import DataAnalyzer
import random

import logging


l_g = logging.getLogger()
l_g.setLevel(logging.DEBUG)

class DataProcesser():
    """
    Just run this file to get processed file.
    Please put the origin file at filepath "../data/chat.txt"

    Note:
    1. Don't import this file at any other file!!!
    2. Loading and processing may take a long time up to 20 mins.

    Functions:
        get_file_primary()
        load_data()
        get_file_fundamental()
        is_bad_line()
        replace_special_field_v1()
    """
    def __init__(self, n_col = 0, col_index = 0, filepath_origin ="./data/chat.txt", filepath_fundamental ="./data/chat_fundamental.txt",
                 filepath_output_primary ="./data/chat_primary.txt", filepath_output_reformated = None):
        """
        Args:
            n_col : total number of columns.
            col_index : the index of the col that contains unexpected tabs.
                index is start from 1.

        Note: col_index is start from 1 !!
        """
        self.filepath_origin = filepath_origin
        self.filepath_output_primary = filepath_output_primary
        self.filepath_output_fundamental = filepath_fundamental
        self.filepath_output_reformated = filepath_output_reformated
        self.n_col = n_col
        self.col_index = col_index-1
        self.data = None

    def get_file_primary(self, list_filter_col = [], is_replace = False):
        """
        Although data can be loaded by pandas, file contains bad lines with nan.
        What's more, we want to clean special fields in texts.
        """
        if os.path.exists(self.filepath_output_primary):
            print("The primary processed file exists")
            if is_replace == True:
                print("Force overwrite.")
            else :
                return
        print("start primary processing")
        self.get_file_fundamental()
        self.load_data(self.filepath_output_fundamental)
        self.drop_bad_line(list_filter_col)

        for i in range(self.data.shape[0]):
            # self.data.iat[i, self.col_index] = self.replace_special_field_v1(self.data.iat[i, self.col_index])
            self.data.iat[i, self.col_index] = self.replace_special_field_v2(self.data.iat[i, self.col_index])

        self.output_data(self.filepath_output_primary)
        print("output file is primary processed.")

    def load_data(self, filepath):
        print("Loading file.")
        data = pd.read_csv(filepath, sep="\t", engine="python",
                           warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header = None)
        self.data = data
        print("Loading finished.")

    def output_data(self, filepath):
        self.data.to_csv(filepath, encoding="utf-8", index=False, sep="\t", header=None)

    def get_file_fundamental(self):
        """
        There may be \t in the col containing long texts.
        So that this file cannot be loaded as .csv file

        This funtion is to remove unexpected tabs in texts.
        Please make sure here's only one col that contains unexpected tabs.

        Args:
            n_col : total number of columns.
            col_index : the index of the col that contains unexpected tabs.
                index is start from 1.

        Note: col_index is start from 1 !!
        """
        if os.path.exists(self.filepath_output_fundamental):
            print("The file after removing unexpected tabs exists.")
            return
        print("start fundamental processing")

        with open(self.filepath_origin, "r", encoding='UTF-8') as file_in, open(
                self.filepath_output_fundamental, "w", encoding='UTF-8') as file_out:
            if self.n_col == 0:
                return
            for line in file_in:
                list_col = line.strip("\n").split("\t")
                str_tmp1 = "\t".join(list_col[0:self.col_index-1])

                if self.n_col != self.col_index:
                    str_given_col = " ".join(list_col[self.col_index:-(self.n_col - self.col_index -1)]).strip()
                    str_given_col = re.sub("['\"', '\0']", " ", str_given_col)
                    str_given_col = re.sub("&nbsp", " ", str_given_col)
                    str_tmp2 = "\t".join(list_col[-(self.n_col-self.col_index -1):])
                    str_res = "\t".join([str_tmp1, str_given_col, str_tmp2])
                else :
                    str_given_col = " ".join(list_col[self.col_index-1:]).strip()
                    str_given_col = re.sub("['\"', '\0']", " ", str_given_col)
                    str_given_col = re.sub("&nbsp", " ", str_given_col)
                    str_res = "\t".join([str_tmp1, str_given_col])
                file_out.write(str_res + "\n")
            file_out.flush()
            os.fsync(file_out.fileno())
        print("output file after removing unexpected tab and being standar csv format.")


    def drop_bad_line(self, list_filter_col):
        for i in range(len(list_filter_col)):
            self.data = self.data[np.array(self.data[[list_filter_col[i]]].notnull()).astype("bool")]

    def replace_special_field_v2(self, sentence):
        """
        特殊字段有：
        1. #E-s[数字x] #E-2[数字x] 等一系列数字—— 表情
        2. [ORDERID_10187709] —— 订单号
        3. [数字x] —— 数字
        4. https://item.jd.com/5898522.html —— 网址
        5. [地址x] —— 地址
        6. [链接x] —— 链接
        7. [金额x] —— 金额
        8. [日期x] —— 日期
        9. [时间x] —— 时间
        10. [站点x] —— 站点
        11. [组织机构x] ——组织机构
        12. [电话x] —— 电话
        13. [姓名x] —— 人名

        对于表情，做法是直接删除。其他用希腊符号替换。
        """
        sentence = re.sub(
            "#E\-[\w]*(抱拳|傲慢|得意|蛋糕|呕吐|闭嘴|礼物|yaoping|柠檬|流泪|怒火|撇嘴|太阳|咒骂|糗|猪猪|足球|磕头|大兵|电话|灯泡|飞鸟|奋斗|高兴|击打|饥饿|咖啡|口罩|骷髅|可乐|疯狂|白眼|阴险|叹气|奸笑|发呆|害羞|飞吻|怒火|悲伤|胜利|生病|弱|可怜|咖啡|酷酷|眩晕|流泪|发抖|难过|右哼哼|惊恐|悲伤|犯困|愤怒|凋谢|哈欠|拥抱|抓狂|鄙视|时间|啤酒|勾引|左哼哼|月亮|偷笑|震惊|惊讶|跳跳|瞌睡|可爱|衰样|好|憨笑|水果|色色|黑线|微笑|流汗|握手|心碎|问号|大哭|亲亲|抠鼻|拜拜|鬼脸|香吻|米饭|花朵|尴尬|擦汗|安慰|委屈|调皮|爱心|我一定尽力为您解答的哦|很棒|鼓掌)+",
            "α", sentence)  ## 匹配 #E-流汗
        sentence = re.sub("#E\-[\w]+\[数字x]", "α", sentence)
        sentence = re.sub("\[ORDERID_[\d]+]", "[订单x]", sentence)
        sentence = re.sub("\[数字x]", "γ", sentence)
        # sentence = re.sub("\[地址x]", "δ", sentence)
        sentence = re.sub("\[链接x]", "ε", sentence)
        # sentence = re.sub("\[金额x]", "ζ", sentence)
        # sentence = re.sub("\[日期x]", "θ", sentence)
        # sentence = re.sub("\[时间x]", "κ", sentence)
        # sentence = re.sub("\[站点x]", "λ", sentence)
        # sentence = re.sub("\[组织机构x]", "μ", sentence)
        # sentence = re.sub("\[电话x]", "ν", sentence)
        # sentence = re.sub("\[姓名x]", "ξ", sentence)
        # sentence = re.sub("\[邮箱x]", "π", sentence)
        # sentence = re.sub("\[身份证号x]", "ρ", sentence)
        # sentence = re.sub("\[商品快照]", "σ", sentence)
        sentence = re.sub("\[表情]", "α", sentence)
        sentence = re.sub(
            "(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?", "ε",
            sentence)
        sentence = re.sub("(http|ftp|https):\/\/ε", "ε", sentence)
        sentence = re.sub("[\d]+.*[\d]+", "γ", sentence)
        sentence = re.sub("【收到不支持的消息类型，暂无法显示】", " ", sentence)

        sentence = re.sub("#E\-[s]*(ν|γ|π|ζ|ρ|α|ε)*", "α", sentence)
        sentence = re.sub("α", " ", sentence)
        sentence = re.sub("ε", "[链接x]", sentence)
        sentence = re.sub("γ", "[数字x]", sentence)

        return sentence

    def replace_special_field_v1(self, sentence):
        """
        特殊字段有：
        1. #E-s[数字x] #E-2[数字x] 等一系列数字—— 表情
        2. [ORDERID_10187709] —— 订单号
        3. [数字x] —— 数字
        4. https://item.jd.com/5898522.html —— 网址
        5. [地址x] —— 地址
        6. [链接x] —— 链接
        7. [金额x] —— 金额
        8. [日期x] —— 日期
        9. [时间x] —— 时间
        10. [站点x] —— 站点
        11. [组织机构x] ——组织机构
        12. [电话x] —— 电话
        13. [姓名x] —— 人名

        对于表情，做法是直接删除。其他用希腊符号替换。
        """
        sentence = re.sub(
            "#E\-[\w]*(抱拳|傲慢|得意|蛋糕|呕吐|闭嘴|礼物|yaoping|柠檬|流泪|怒火|撇嘴|太阳|咒骂|糗|猪猪|足球|磕头|大兵|电话|灯泡|飞鸟|奋斗|高兴|击打|饥饿|咖啡|口罩|骷髅|可乐|疯狂|白眼|阴险|叹气|奸笑|发呆|害羞|飞吻|怒火|悲伤|胜利|生病|弱|可怜|咖啡|酷酷|眩晕|流泪|发抖|难过|右哼哼|惊恐|悲伤|犯困|愤怒|凋谢|哈欠|拥抱|抓狂|鄙视|时间|啤酒|勾引|左哼哼|月亮|偷笑|震惊|惊讶|跳跳|瞌睡|可爱|衰样|好|憨笑|水果|色色|黑线|微笑|流汗|握手|心碎|问号|大哭|亲亲|抠鼻|拜拜|鬼脸|香吻|米饭|花朵|尴尬|擦汗|安慰|委屈|调皮|爱心|我一定尽力为您解答的哦|很棒|鼓掌)+",
            "α", sentence)  ## 匹配 #E-流汗
        sentence = re.sub("#E\-[\w]+\[数字x]", "α", sentence)
        sentence = re.sub("\[ORDERID_[\d]+]", "β", sentence)
        sentence = re.sub("\[数字x]", "γ", sentence)
        sentence = re.sub("\[地址x]", "δ", sentence)
        sentence = re.sub("\[链接x]", "ε", sentence)
        sentence = re.sub("\[金额x]", "ζ", sentence)
        sentence = re.sub("\[日期x]", "θ", sentence)
        sentence = re.sub("\[时间x]", "κ", sentence)
        sentence = re.sub("\[站点x]", "λ", sentence)
        sentence = re.sub("\[组织机构x]", "μ", sentence)
        sentence = re.sub("\[电话x]", "ν", sentence)
        sentence = re.sub("\[姓名x]", "ξ", sentence)
        sentence = re.sub("\[邮箱x]", "π", sentence)
        sentence = re.sub("\[身份证号x]", "ρ", sentence)
        sentence = re.sub("\[商品快照]", "σ", sentence)
        sentence = re.sub("\[表情]", "α", sentence)
        sentence = re.sub(
            "(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?", "ε",
            sentence)
        sentence = re.sub("(http|ftp|https):\/\/ε", "ε", sentence)
        sentence = re.sub("[\d]+.*[\d]+", "γ", sentence)
        sentence = re.sub("【收到不支持的消息类型，暂无法显示】", " ", sentence)

        sentence = re.sub("#E\-[s]*(ν|γ|π|ζ|ρ|α|ε)*", "α", sentence)
        sentence = re.sub("α", " ", sentence)

        return sentence

    def get_file_middle(self, k_per):
        """
        This function is used to:
        1. unite QQQA into QA format
        2. drop first A if a session starts from A.
        3. drop lines with empty text
        4. drop lines whose length is more than 100
        """

        import random
        print("start reformating.")
        try :
            if self.data == None:
                self.load_data(self.filepath_output_primary)
        except :
            pass

        ## drop lines with empty text
        data_tmp = pd.Series(self.data[[6]].replace(to_replace=r'^\s*$', value=np.nan, regex=True).values.flatten())
        self.data = self.data[np.array(data_tmp.notnull()).astype(bool)]

        ## drop lines whose length is more than 100
        list_sentence_length = [len(x) for x in self.data[[6]].values.flatten()]
        self.data = self.data[np.array(list_sentence_length) < 100]

        da = DataAnalyzer()
        cnt_session, x_session_ptr, x_session_length = da.get_session_info(self.data)

        filepath_output = None
        if self.filepath_output_reformated != None:
            filepath_output = self.filepath_output_reformated
        else :
            filepath_output = "./data/chat_" + str(k_per) + "per.txt"

        with open(filepath_output, "w", encoding='UTF-8') as f_out:
            for i in range(cnt_session + 1):
                if random.uniform(0, 1) <= (k_per/100):
                    sentence = ""
                    is_first = True
                    for j in range(x_session_ptr[i], x_session_ptr[i + 1]):
                        ## 保证是QAQA的形式，而不是AQAQAQ

                        if is_first and (self.data.iat[j, 2] == "1"):
                            continue
                        is_first = False
                        ## 如果QQQAQA的第一个Q那么先输出这一行的0-5列，再存储句子
                        ## 如果是中继的QQ，则加上
                        ## 如果是末尾的，则输出sentence，并重置

                        ## 如果是QA的Q，则直接输出行

                        ## 对于下一个不存在的情况，根据sentence是否存在看是输出sentence或是整行

                        if sentence == "" and j + 1 < x_session_ptr[i + 1] and self.data.iat[j, 2] == self.data.iat[j + 1, 2]:
                            sentence += self.data.iat[j, 6]
                        elif sentence != "" and j + 1 < x_session_ptr[i + 1] and self.data.iat[j, 2] == self.data.iat[j + 1, 2]:
                            sentence += " " + self.data.iat[j, 6]
                        elif sentence != "" and j + 1 < x_session_ptr[i + 1] and self.data.iat[j, 2] != self.data.iat[j + 1, 2]:
                            sentence += " " + self.data.iat[j, 6]
                            f_out.write(str(self.data.iat[j, 0]) + "\t" + str(self.data.iat[j, 1]) + "\t" +
                                        str(self.data.iat[j, 2]) + "\t" + str(self.data.iat[j, 3]) + "\t" +
                                        str(self.data.iat[j, 4]) + "\t" + str(self.data.iat[j, 5]) + "\t" + sentence + "\n")
                            sentence = ""
                        elif sentence == "" and j + 1 < x_session_ptr[i + 1] and self.data.iat[j, 2] != self.data.iat[j + 1, 2]:
                            f_out.write(str(self.data.iat[j, 0]) + "\t" + str(self.data.iat[j, 1]) + "\t" + str(self.data.iat[j, 2])
                                        + "\t" + str(self.data.iat[j, 3]) + "\t" + str(self.data.iat[j, 4]) + "\t" + str(self.data.iat[j, 5])
                                        + "\t" + self.data.iat[j, 6] + "\n")
                        elif sentence == "" and j + 1 == x_session_ptr[i + 1]:
                            f_out.write(str(self.data.iat[j, 0]) + "\t" + str(self.data.iat[j, 1]) + "\t" +
                                        str(self.data.iat[j, 2]) + "\t" + str(self.data.iat[j, 3]) + "\t" +
                                        str(self.data.iat[j, 4]) + "\t" + str(self.data.iat[j, 5]) + "\t" +
                                        self.data.iat[j, 6] + "\n")
                        elif sentence != "" and j + 1 == x_session_ptr[i + 1]:
                            sentence += " " + self.data.iat[j, 6]
                            f_out.write(str(self.data.iat[j, 0]) + "\t" + str(self.data.iat[j, 1]) + "\t" +
                                        str(self.data.iat[j, 2]) + "\t" + str(self.data.iat[j, 3]) + "\t" +
                                        str(self.data.iat[j, 4]) + "\t" + str(self.data.iat[j, 5]) + "\t" + sentence + "\n")
                            sentence = ""
                else:
                    continue

        print("output file after reformating.")


    @classmethod
    def generate_train_file_bert_single(cls):
        """
        The format that bert needs is "q a label"
        """
        filepath_input = "./data/chat_100per.txt"
        data_total = pd.read_csv(filepath_input, sep = "\t")
        logging.info("Total data size: " + str(data_total.shape[0]))

        filepath_output = "./code/bert/data/train.tsv"
        with open(filepath_output, "w", encoding="utf-8") as f_out:
            for i in range(data_total.shape[0]):
                if i == data_total.shape[0]-1:
                    break
                if data_total.iat[i, 2] == 0 and data_total.iat[i+1, 2] == 1 and \
                    data_total.iat[i, 0] == data_total.iat[i+1, 0]:
                    q = data_total.iat[i, 6]
                    true_a = data_total.iat[i+1, 6]
                    f_out.write(q + "\t" + true_a + "\t1\n")

                    flag = True
                    while flag:
                        index_false_a = random.randint(0, data_total.shape[0]-5)
                        while data_total.iat[index_false_a, 2] != 1:
                            index_false_a += 1

                    false_a = data_total.iat[index_false_a, 6]
                    f_out.write(q + "\t" + false_a + "\t0\n")
                if i % 10000 == 0:
                    logging.info("Finished {0} sentences.".format(i))

            # logging.info("Generating Positive samples: " + str(len(list_true_a)))
            # logging.info("Generating Negetive samples: " + str(len(list_false_a)))

            # for i in range(len(list_q)):
            #     f_out.write(list_q[i] + "\t" + list_true_a + "\t1\n")
            #     f_out.write(list_q[i] + "\t" + list_false_a + "\t0\n")
        logging.info("output candidate file as bert format to:" + filepath_output)




# dp = DataProcesser(7, 7)
# dp.get_file_primary([0, 1, 2, 3, 4], is_replace= False)
#
# # ## adjust this function arg "k_per" to select k percentage data
# dp.get_file_middle(5)
#
# ## get bert train file
# DataProcesser.generate_train_file_bert_single()


