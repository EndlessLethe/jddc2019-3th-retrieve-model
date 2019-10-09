import numpy as np
import time
from gensim import similarities
from jieba_seg import JiebaSeg

class UnsupervisedReranker():
    def __init__(self):
        self.seg_jieba = JiebaSeg()

    def get_word_vector(self, s1,s2):
        """
        :param s1: 句子1
        :param s2: 句子2
        :return: 返回句子的余弦相似度
        """
        # 分词
        # cut1 = self.seg_jieba.cut(s1, True)
        # cut2 = self.seg_jieba.cut(s2, True)
        cut1 = self.seg_jieba.cut(s1, False)
        cut2 = self.seg_jieba.cut(s2, False)
        list_word1 = (','.join(cut1)).split(',')
        list_word2 = (','.join(cut2)).split(',')

        # 列出所有的词,取并集
        key_word = list(set(list_word1 + list_word2))
        # 给定形状和类型的用0填充的矩阵存储向量
        word_vector1 = np.zeros(len(key_word))
        word_vector2 = np.zeros(len(key_word))

        # 计算词频
        # 依次确定向量的每个位置的值
        for i in range(len(key_word)):
            # 遍历key_word中每个词在句子中的出现次数
            for j in range(len(list_word1)):
                if key_word[i] == list_word1[j]:
                    word_vector1[i] += 1
            for k in range(len(list_word2)):
                if key_word[i] == list_word2[k]:
                    word_vector2[i] += 1

        # 输出向量
    #     print(word_vector1)
    #     print(word_vector2)
        return word_vector1, word_vector2

    def cos_dist(self, s1, s2):
        """
        需要调用get_word_vector()得到向量化表示

        :param vec1:
        :param vec2:
        :return: 返回两个句子的余弦相似度
        """
        vec1, vec2 = self.get_word_vector(s1, s2)
        dist1= float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        return dist1

    def get_first_sentence(self, list_list_kanswer, data):
        if list_list_kanswer[0][0] + 1 >= data.shape[0]:
            return 1
        return 0

    def get_center_sentence(self, list_list_kanswer, data, k):
        x_sim = np.zeros((k, k))
        for i in range(k):
            if list_list_kanswer[i][0] + 1 >= data.shape[0]:
                continue
            if len(data.iat[list_list_kanswer[i][0] + 1, 0]) <= 7: continue
            for j in range(k):
                if list_list_kanswer[j][0] + 1 >= data.shape[0]:
                    continue
                if i != j:
                    x_sim[i][j] = self.cos_dist(data.iat[list_list_kanswer[i][0] + 1, 0],
                                                data.iat[list_list_kanswer[j][0] + 1, 0])
        #             print("i j: " + str(i) + " " + str(j) + " " + str(cos_dist(data_chat.iat[list_list_kanswer[i][0]+1, 6], data_chat.iat[list_list_kanswer[j][0]+1, 6])))

        # print(x_sim)

        x_sum = np.zeros((k,))
        for i in range(k):
            x_sum[i] = x_sim[i].sum()
        n_result = np.argmax(x_sum)
        return n_result

    def get_maxlen_sentence(self, list_list_kanswer, data, k):
        x_len = np.zeros((k, ))
        for i in range(k):
            x_len[i] = len(data.iat[list_list_kanswer[i][0] + 1, 0])
        n_result = np.argmax(x_len)
        return n_result


    def similarity(self, list_tuple_kanswer, data, k):
        n_result = self.get_center_sentence(list_tuple_kanswer, data, k)
        # n_result = self.get_maxlen_sentence(list_tuple_kanswer, data, k)
        # n_result = self.get_first_sentence(list_tuple_kanswer, data)
        return n_result



