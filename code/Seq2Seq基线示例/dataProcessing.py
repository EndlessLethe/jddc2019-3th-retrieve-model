# -*- coding: utf-8 -*-
"""
Created on Wed Apr  10 17:08:59 2018
@author: Simon
"""

import codecs
import logging
import random
import os

def data_processing():
    with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/questions.txt", mode="w", encoding="utf-8") as wfquestion:
        with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/answers.txt", mode="w", encoding="utf-8") as wfanswer:
            try:
                wfquestion.truncate()
                wfanswer.truncate()
            except Exception as e:
                logging.info("data_processing:clear data_processing.txt error:", e)
            finally:
                wfquestion.close()
                wfanswer.close()
    
    question = ''
    answer = ''
    QAQAQ = ''
    countQuestion = 0
    countAnswer = 0
    sessionId = "00029c51f92e8f34250d6af329c9a8df"  #第一行的sessionID
    with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/chat.txt", mode = 'r', encoding = "utf-8") as rf:
        try:
            line = rf.readline()
            while line:
                splitline = line.strip('\r\n').split("\t")
                if sessionId == splitline[0]:
                    with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/questions.txt", mode="a", encoding="utf-8") as wf_question:
                        with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/answers.txt", mode="a", encoding="utf-8") as wf_answer:
                            try:
                                if splitline[2] == '0':
                                    if countQuestion == 3 and countAnswer == 2 :
                                        wf_question.write(QAQAQ + "\n")
                                        wf_answer.write(answer + "\n")
                                        question = ''
                                        answer = ''
                                        QAQAQ = ''
                                        countQuestion = 0
                                        countAnswer = 0
                                        
                                    if answer != '':
                                        #answer = answer.strip(',')
                                        #wf_question.write(answer)
                                        QAQAQ = QAQAQ + answer
                                        answer = ''
                                        countAnswer = countAnswer + 1    
                                    question = question + splitline[6] + ','
                                        
                                elif splitline[2] == '1':
                                    if question != '':
                                        #question = question.strip(',')
                                        #wf_question.write(question)
                                        QAQAQ = QAQAQ + question
                                        question = ''
                                        countQuestion = countQuestion + 1                                   
                                    answer = answer + splitline[6] + ','
                                    
                            except Exception as e:
                                logging.error("data_processing:write into chatmasked_user failure", e)
                            finally:
                                wf_question.close()
                                wf_answer.close()
                                
                else:
                    sessionId = splitline[0]
                    question = ''
                    answer = ''
                    QAQAQ = ''
                    countQuestion = 0
                    countAnswer = 0
                    continue
                
                line = rf.readline()
                
        except Exception as e:
            logging.error("data_processing: data processing failure!", e)
        finally:
            rf.close()
   
def cutDataToTrainDevBy91():
    randomList = []
    with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/devQuestions.txt", mode="w", encoding="utf-8") as wf_devQuestion:
        with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/devAnswers.txt", mode="w", encoding="utf-8") as wf_devAnswer:
            with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/trainQuestions.txt", mode="w", encoding="utf-8") as wf_trainQuestion:
                with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/trainAnswers.txt", mode="w", encoding="utf-8") as wf_trainAnswer:
                    try:
                        wf_devQuestion.truncate()
                        wf_devAnswer.truncate()
                        wf_trainQuestion.truncate()
                        wf_trainAnswer.truncate()
                    except Exception as e:
                        logging.info("data_processing:clear data_processing.txt error:", e)
                    finally:
                        wf_devQuestion.close()
                        wf_devAnswer.close()    
    
    with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/questions.txt", mode = 'r', encoding = "utf-8") as rf_question:
        with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/answers.txt", mode = 'r', encoding = "utf-8") as rf_answer:
            try:
                questionLines = rf_question.readlines()
                answerLines = rf_answer.readlines()
                #trainset的十分之一的数据集作为devset
                randomList = random.sample(range(len(questionLines)-1), int(len(questionLines)/10))
                with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/devQuestions.txt", mode = 'a', encoding = "utf-8") as wf_devQuestion:
                    with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/devAnswers.txt", mode = 'a', encoding = "utf-8") as wf_devAnswer:
                        try:
                            for i in randomList:
                                wf_devQuestion.write(questionLines[i])
                                wf_devAnswer.write(answerLines[i])
                        except Exception as e:
                            logging.error("cutDataToTrainDevBy91: failure", e)
                        finally:
                            wf_devQuestion.close()
                            wf_devAnswer.close()
                
            except Exception as e:
                logging.error("cutDataToTrainDevBy91: failure", e)
            finally:
                rf_question.close()
                rf_answer.close()
                
    with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/questions.txt", mode = 'r', encoding="utf-8") as rf_question:
        with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/answers.txt", mode='r',encoding="utf-8") as rf_answer:
            questions = rf_question.readlines()
            answers = rf_answer.readlines()
            with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/trainQuestions.txt",mode='a',encoding="utf-8") as wf_question:
                with codecs.open("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/trainAnswers.txt",mode='a',encoding="utf-8") as wf_answer:
                    for i in range(len(questions)):
                        if i not in randomList:
                            wf_question.write(questions[i])
                    for i in range(len(answers)):
                        if i not in randomList:
                            wf_answer.write(answers[i])
                            
                    rf_question.close()
                    rf_answer.close()
                    wf_question.close()
                    wf_answer.close()
    
                            
    os.remove("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/questions.txt")
    os.remove("C:/Users/Simon/Desktop/智能对话大赛/开源基线/算法大赛数据集/脱敏第二版日志-3/算法大赛基线数据集QAQAQ+A/answers.txt")
                            
                
if __name__ == "__main__": 
    data_processing()
    cutDataToTrainDevBy91()
 