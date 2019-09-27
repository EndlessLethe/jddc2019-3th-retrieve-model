import time
from gensim import similarities
from embedding_model_loader import EmbeddingModelLoader
from unsupervised_reranker import UnsupervisedReranker
import pandas as pd
from jieba_seg import JiebaSeg

class RunModel():
    """
    This Class is used to connect all components, such as embedding part for training,
    and search part, rerank part for predicting.
    """
    def __init__(self, filepath_input):
        self.filepath_input = filepath_input
        self.dictionary = None
        self.index = None
        self.data = None
        self.model = None
        self.seg_jieba = JiebaSeg()

    def load_data(self):
        data_chat = pd.read_csv(self.filepath_input, sep="\t", engine="python",
                                warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
        # return data_chat[[0]]
        return data_chat[[6]]

    def fit(self, num_topics = 40):
        self.data = self.load_data()
        model = EmbeddingModelLoader()
        ## could change the model used
        # self.dictionary, self.model, self.corpus_embedding = model.tfidf_fit(self.data)
        # self.dictionary, self.model, self.corpus_embedding = model.lsi_fit(self.data, num_topics)
        self.dictionary, self.model, self.corpus_embedding = model.lda_fit(self.data, num_topics)

        self.index = self.get_index()

    def get_index(self):
        # self.index = similarities.MatrixSimilarity(corpus_tfidf, num_features=len(self.dictionary))
        index = similarities.Similarity("out/", self.corpus_embedding, len(self.dictionary))
        return index

    ## 先使用sentence2vec将需要匹配的句子传进去
    def sentence2vec(self, sentence):
        # list_word = list(self.seg_jieba.cut(sentence, True))
        list_word = list(self.seg_jieba.cut(sentence, False))

        vec_bow = self.dictionary.doc2bow(list_word)
        return self.model[vec_bow]

    def get_topk_answer(self, sentence, k=15):
        """求最相似的句子"""
        sentence_vec = self.sentence2vec(sentence)
        sims = self.index[sentence_vec]

        # 按相似度降序排序
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        top_k = sim_sort[0:k]

        return top_k

    def predict(self, session_list, session_length, session_text, filepath_result, k = 15):
        ur = UnsupervisedReranker()

        with open(filepath_result, "w", encoding='utf-8') as f_out:
            cnt = 0
            for i in range(len(session_list)):
                f_out.write("<session " + session_list[i] + ">\n")
                for j in range(session_length[i]):
                    list_list_kanswer = self.get_topk_answer(session_text[cnt], k)
                    f_out.write(ur.similarity(list_list_kanswer, self.data, k)[1] + "\n")
                    cnt += 1
                f_out.write("</session " + session_list[i] + ">\n\n")

