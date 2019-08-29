#encoding=utf-8

from cutWords import *
from fileObject import FileObj
from sentenceSimilarity import SentenceSimilarity
from sentence import Sentence

if __name__ == '__main__':
    # 读入训练集
    file_obj = FileObj(r"dataSet/trainQuestions.txt")  
    train_sentences = file_obj.read_lines()
   

    # 读入测试集
    file_obj = FileObj(r"dataSet/devQuestions.txt")   
    test_sentences = file_obj.read_lines()


    # 分词工具，基于jieba分词，并去除停用词
    seg = Seg()

    # 训练模型
    ss = SentenceSimilarity(seg)
    ss.set_sentences(train_sentences)
    ss.TfidfModel()         # tfidf模型

    # 测试集
    right_count = 0
    
    file_result=open('dataSet/result.txt','w')
    with open("dataSet/trainAnswers.txt",'r',encoding = 'utf-8') as file_answer:
        line = file_answer.readlines()
           
    for i in range(0,len(test_sentences)):
        top_15 = ss.similarity(test_sentences[i])
        
        for j in range(0,len(top_15)):
            answer_index=top_15[j][0]
            answer=line[answer_index]
            file_result.write(str(top_15[j][1])+'\t'+str(answer))
        file_result.write("\n")
        
    file_result.close() 
    file_answer.close()
    