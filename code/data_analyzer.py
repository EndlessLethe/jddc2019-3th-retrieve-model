import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataAnalyzer():

    def __init__(self, filepath_input = None):
        """
        This class provide 2 ways to use:
        1. import as moudle
        2. just run this .py to see what exists in data directly.
            (At last of this file, here's some example codes)

        self.data is usually None to decrease the usage of space.
        It is used only when user just want to run DataAnalyzer only and call its function conveniently.
        """
        self.data = None
        self.filepath_input = filepath_input
        self.x_session_length = None
        self.x_session_ptr = None
        self.cnt_session = None

    def load_data(self):
        data_chat = pd.read_csv(self.filepath_input, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
        # return data_chat[[0]]
        self.data = data_chat

    def plot_distribution(self, x_data):
        from collections import Counter
        dict_data = Counter(np.sort(x_data))
        # print(dict_data)

        X = np.array(list(dict_data.keys())).reshape(-1, 1)  # numbers of one same feature value
        Y = np.array(list(dict_data.values())).reshape(-1, 1)  # count

        plt.plot(X, Y)
        plt.show()

        X = np.log10(np.array(list(dict_data.keys()))).reshape(-1, 1)  # numbers of one same feature value
        Y = np.log10(list(dict_data.values())).reshape(-1, 1)  # count

        plt.plot(X, Y)
        plt.show()

    def plot_extrame(self, x_data):

        plt.boxplot(x_data)
        plt.show()

        ## Q1为数据占25%的数据值范围
        Q1 = np.percentile(x_data, 25)

        ## Q3为数据占75%的数据范围
        Q3 = np.percentile(x_data, 75)
        IQR = Q3 - Q1

        ## 正常值的范围
        outlier_step = 0 * IQR
        list_standard = [x for x in x_data if x < Q3 + outlier_step or x > Q1 - outlier_step]

        print("IQO = 0, 正常值的范围：", Q3 + outlier_step, Q1 - outlier_step)
        print("正常点的数量", len(list_standard))

        ## soft 异常值的范围
        outlier_step = 1.5 * IQR
        list_outlier = [x for x in x_data if x > Q3 + outlier_step or x < Q1 - outlier_step]

        print("IQO = 1.5, 异常值临界点：", Q3 + outlier_step, Q1 - outlier_step)
        print("异常点的数量", len(list_outlier))

        ## hard 异常值的范围
        ## 这个hard异常值是测试得到的，发现3 * IQR有1w+异常值，但5 * IQR，比较合适
        outlier_step = 5 * IQR
        list_outlier = [x for x in x_data if x > Q3 + outlier_step or x < Q1 - outlier_step]

        print("IQO = 5, 异常值临界点：", Q3 + outlier_step, Q1 - outlier_step)
        print("极端异常点的数量", len(list_outlier))

    def get_session_info(self, data):
        """
        访问data_chat的数据的步骤：
        1. 先利用session_index获取到index: index = x_session_ptr[session_index]
        2. 在通过.iat确定访问第i列
        """
        set_session = set()
        set_user = set()

        ## 对于每个session，我们统计两个量length和ptr
        ## length是每个session对话的长度，ptr是每个对话在data中的起始位置。
        x_session_length = np.zeros((2000000,), dtype="int")
        x_session_ptr = np.zeros((2000000,), dtype="int")
        cnt_session = 0

        # max_session_length = -1
        # max_user_id = -1
        # max_session_id = -1
        # max_session_index = -1

        for i in range(data.shape[0]):
            if not data.iat[i, 0] in set_session:
                if i != 0:
                    x_session_length[cnt_session] = cnt_session_length
                    cnt_session += 1
                x_session_ptr[cnt_session] = i
                cnt_session_length = 1
                set_session.add(data.iat[i, 0])
                set_user.add(data.iat[i, 1])
            else:
                cnt_session_length += 1
                # if cnt_session_length > max_session_length:
                #     max_session_length = cnt_session_length
                    # max_user_id = self.data.iat[i, 1]
                    # max_session_id = self.data.iat[i, 0]
                    # max_session_index = cnt_session
        x_session_length[cnt_session] = cnt_session_length
        x_session_ptr[cnt_session + 1] = data.shape[0]

        self.x_session_length = x_session_length[0:cnt_session + 1]
        self.x_session_ptr = x_session_ptr[0:cnt_session + 2]
        self.cnt_session = cnt_session

        return cnt_session, x_session_ptr, x_session_length
        # print(len(set_session), len(set_user))
        # print(max_session_index, max_user_id, max_session_id, max_session_length)

    def show_session_info(self):
        self.plot_distribution(self.x_session_length)
        self.plot_extrame(self.x_session_length)

    def print_session_top_len(self, data):
        ## 找到length对应的session index
        x_session_length_argsort = np.argsort(self.x_session_length)
        print(self.x_session_length[x_session_length_argsort[-1]])  ## 应该为314
        print(x_session_length_argsort[-1])  ## 应该为461218
        print(self.x_session_ptr[x_session_length_argsort[-1]])  ## 应该为9242002

        ## 输出前10个异常对话，观察特点
        for i in range(1, 11):
            print()
            ptr_now = self.x_session_ptr[x_session_length_argsort[-i]]
            ptr_end = self.x_session_ptr[x_session_length_argsort[-i] + 1]  ## 最后一个对话不是异常的，所以不会out of index
            print("i == " + str(i), ptr_end - ptr_now)
            print()
            while (ptr_now < ptr_end):
                print(int(data.iat[ptr_now, 2]), data.iat[ptr_now, 6])
                ptr_now += 1

# da = DataAnalyzer("../data/chat_1per.txt")
# da.load_data()
# # # da.get_session_info(da.data)
# # # # print(da.data.iat[da.x_session_ptr[da.cnt_session], 0]) ## 应该是 fffd0cf8-b5df-4d7e-baaf-91776fac2024
# # # print(da.x_session_length)
# # # da.show_session_info()
# #
# list_sentence_length = [len(x) for x in da.data[[6]].values.flatten()]
# da.plot_distribution(np.array(list_sentence_length))
# da.plot_extrame(np.array(list_sentence_length))