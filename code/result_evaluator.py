from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jieba_seg import JiebaSeg

class ResultEvaluator():
    """
    读入的两个文件都应该是一个一行一个语句的形式。
    会跳过首字母为"\n"或"<"的行
    """
    def __init__(self, filepath_result = "./out/test.txt", filepath_test ="./data/"):
        self.filepath_result = filepath_result
        self.filepath_test = filepath_test

    def eval_result(self):
        list_predict, list_answer = self.load_data()
        n_eval = self.get_bleu(list_predict, list_answer)
        return n_eval

    def load_data(self):
        file_result = open(self.filepath_result, "r", encoding="utf-8")
        file_answer = open(self.filepath_test, "r", encoding="utf-8")
        list_tmp1 = file_result.readlines()
        list_tmp2 = file_answer.readlines()
        list_predict = []
        list_answer = []
        for i in range(len(list_tmp1)):
            if list_tmp1[i][0] == "\n" or list_tmp1[i][0] == "<":
                continue
            else :
                list_predict.append(list_tmp1[i])
                list_answer.append(list_tmp2[i])
        return list_predict, list_answer

    def get_bleu(self, list_predict, list_answer):
        n_sum = 0
        seg_jieba = JiebaSeg()

        smooth = SmoothingFunction()
        for i in range(len(list_predict)):
            ## 下面注释掉的写法有错误
            # n_eval = sentence_bleu(list_answer[i], list_predict[i], smoothing_function=smooth.method1)
            n_eval = sentence_bleu([seg_jieba.cut(list_answer[i])], seg_jieba.cut(list_predict[i]), smoothing_function=smooth.method1)
            n_sum +=  n_eval
            # print(n_eval, list_answer[i], list_predict[i])
        return n_sum / len(list_predict)

# re = ResultEvaluator()
# print(re.get_bleu(["好的，祝您工作愉快，再见!"], ["好的，祝您工作愉快，再见!"]))


# reference=['The', 'new', 'translator', 'will', 'stand', 'on', 'the', 'exhibition', 'on', 'behalf', 'of', 'the', 'four', 'times', 'group', 'at', 'the', 'exhibition', 'We', 'will', 'introduce', 'the', 'new', 'star`s', 'business', 'the', 'advantages', 'and', 'the', 'successful', 'cases', 'so', 'that', 'you', 'can', 'understand', 'the', 'new', 'translator', 'more', 'comprehensively', 'We', 'have', 'a', 'stable', 'full-time', 'international', 'team', 'that', 'ensures', 'punctual', 'efficient', 'translation', 'and', 'dubbing', 'and', 'provides', 'a', 'full', 'range', 'of', 'control', 'through', 'the', 'perfect', 'quality', 'control', 'and', 'project', 'management', 'system', 'providing', 'a', 'one-stop', 'service', 'for', 'translation', 'dubbing', 'subtitle', 'production', 'post', 'production', 'broadcasting', 'and', 'ratings', 'surveys'],['The', 'new', 'translator', 'star', 'will', 'represent', 'sida', 'times', 'group', 'in', 'the', 'exhibition', 'when', 'we', 'will', 'introduce', 'the', 'new', 'translator', 'star`s', 'business', 'advantages', 'successful', 'cases', 'and', 'other', 'dimensions', 'so', 'that', 'you', 'can', 'have', 'a', 'more', 'comprehensive', 'understanding', 'of', 'the', 'new', 'translator', 'star', 'We', 'have', 'a', 'stable', 'full-time', 'international', 'team', 'which', 'can', 'ensure', 'timely', 'and', 'efficient', 'translation', 'and', 'dubbing', 'Through', 'perfect', 'quality', 'control', 'and', 'project', 'management', 'system', 'we', 'provide', 'translation', 'dubbing', 'subtitle', 'production', 'post-production', 'broadcasting', 'and', 'rating', 'survey']
# candidate_baidu=['New', 'Transtar', 'will', 'stand', 'on', 'the', 'exhibition', 'on', 'behalf', 'of', 'the', 'four', 'times', 'group', 'at', 'the', 'exhibition', 'We', 'will', 'introduce', 'the', 'new', 'star`s', 'business', 'the', 'advantages', 'and', 'the', 'successful', 'cases', 'so', 'that', 'you', 'can', 'understand', 'the', 'new', 'translator', 'more', 'comprehensively', 'We', 'have', 'a', 'stable', 'full-time', 'international', 'team', 'that', 'ensures', 'punctual', 'efficient', 'translation', 'and', 'dubbing', 'and', 'provides', 'a', 'full', 'range', 'of', 'control', 'through', 'the', 'perfect', 'quality', 'control', 'and', 'project', 'management', 'system', 'providing', 'a', 'one-stop', 'service', 'for', 'translation', 'dubbing', 'subtitle', 'production', 'streamlined', 'and', 'developed', 'quality', 'control', 'and', 'project', 'management', 'system']
# n_eval = sentence_bleu(reference, candidate_baidu)
# print(n_eval)
#
# str_test2 = "审核时效：18:00前提交的服务单，当日24:00审核完毕，18:00后提交的服务单，次日24:00前审核完毕（超时系统自动审核通过说明：当天18:00-次日18:00期间提交的服务单，如商家未进行审核，系统会在次日24:00前自动审核通过!@@@!已经提交!@@@!会尽快审核哦<sep>好的，已经提交了哦!@@@!审核时效：18:00前提交的服务单，当日24:00审核完毕，18:00后提交的服务单，次日24:00前审核完毕（超时系统自动审核通过说明：当天18:00-次日18:00期间提交的服务单，如商家未进行审核，系统会在次日24:00前自动审核通过!@@@!还请您保持手机畅通哦<sep>NULL"
# str_test1 = "好的，祝您工作愉快，再见!好的，祝您工作愉快，再见!好的，祝您工作愉快，再见!好的，祝您工作愉快，再见!"
# str_test3 = "以后不能家门口"
# str_test4 = "好的，祝您工作愉快，再见!"
# list_test = list(jieba.cut(str_test))
# list_result = list(jieba.cut(str_result))
#
#
# n_eval = sentence_bleu(str_test4, str_test2) ## (ref, can)
# print(n_eval)
# n_eval = sentence_bleu(str_test2, str_test4)
# print(n_eval)
# n_eval = sentence_bleu([list(jieba.cut(str_test4))], list(jieba.cut(str_test4)), smoothing_function=smooth.method1)
# print(n_eval)

