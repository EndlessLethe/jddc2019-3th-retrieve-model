import jieba

class JiebaSeg() :
    def __init__(self, filepath_stopwords = "./data/stoplist.txt"):
        self.stopwords = [line.strip() for line in open(filepath_stopwords, 'r', encoding='utf-8').readlines()]


    def cut(self, sentence, stopwords = False):
        seg_list = jieba.cut(sentence)  # 切词

        if stopwords:
            results = []
            for seg in seg_list:
                if seg in self.stopwords:
                    continue  # 去除停用词
                results.append(seg)
        else :
            results = list(seg_list)
        return results