import pandas as pd
import os
import re
import numpy as np

class DataProcesser():
    """
    Just run function get_file_primary_processed() to get processed file.

    Functions:
        get_file_primary_processed()
        load_data()
        get_file_standar_csv_format()
        is_bad_line()
        replace_special_field()
    """
    def __init__(self, n_col = 0, col_index = 0, filepath_origin ="../data/chat.txt", filepath_output_primary ="../data/chat_primary.txt",
                 filepath_fundamental ="../data/chat_fundamental.txt"):
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
        self.n_col = n_col
        self.col_index = col_index-1
        self.data = None

    def get_file_primary_processed(self, list_filter_col = []):
        if os.path.exists(self.filepath_output_primary):
            print("The primary processed file exists")
            return
        self.get_file_standar_csv_format()
        self.load_data()
        self.drop_bad_line(list_filter_col)

        for i in range(self.data.shape[0]):
            self.data.iat[i, self.col_index] = self.replace_special_field(self.data.iat[i, self.col_index])

        self.data.to_csv(self.filepath_output_primary, encoding="utf-8", index = False, sep = "\t", header = None)
        print("output file is primary processed.")

    def load_data(self):
        data = pd.read_csv(self.filepath_output_fundamental, sep="\t", engine="python",
                           warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header = None)
        self.data = data

    def get_file_standar_csv_format(self):
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
        with open(self.filepath_origin, "r", encoding='UTF-8') as file_in, open(
                self.filepath_output_fundamental, "w", encoding='UTF-8') as file_out:
            if self.n_col == 0:
                return
            for line in file_in:
                list_col = line.strip("\n").split("\t")
                str_tmp1 = "\t".join(list_col[0:self.col_index])

                if self.n_col != self.col_index+1:
                    str_given_col = " ".join(list_col[self.col_index:-(self.n_col - self.col_index -1)]).strip()
                    str_given_col = re.sub("['\"', '\0']", " ", str_given_col)
                    str_given_col = re.sub("&nbsp", " ", str_given_col)
                    str_tmp2 = "\t".join(list_col[-(self.n_col-self.col_index -1):])
                    str_res = "\t".join([str_tmp1, str_given_col, str_tmp2])
                else :
                    str_given_col = " ".join(list_col[self.col_index:]).strip()
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

    def replace_special_field(self, sentence):
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

    # def


dp = DataProcesser(7, 7)
dp.get_file_primary_processed([0, 1, 2, 3, 4])
# print(dp.replace_special_field("小妹正在火速为您查询，还请您稍等一下呢，谢谢#E-s[数字x]"))
