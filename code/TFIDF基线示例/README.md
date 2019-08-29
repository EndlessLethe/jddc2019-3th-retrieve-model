# JD Dialog Challenge

##关于这个代码

这个代码是JD.com关于JD Dialog Challenge大赛向广大参赛选手们提供的开源基线代码，选手们可以在此代码上做出修改和优化，也可以不做参考。

这个代码采用的是tfidf模型，主要思路是：首先计算用户的问题与问题库中的问题的相似度并选出top15的相似问题，然后去问题库对应的答案库中找出这15个问题对应的答案，
以此作为回答用户问题的候选答案。代码参考https://github.com/WenDesi/sentenceSimilarity，运行于python3.6环境下。

dataProcessing.py：数据预处理的文件。
fileObject.py：封装读文件操作。按行读取文件内容，并返回一个List。
cutWords.py：基于jieba分词工具，加入去除停用词做了一个封装，停用词是dataSet目录下的stopword.txt。
sentence.py：把对句子的所有处理做了一个封装包括：对句子进行切词、获取切词之后的列表、获取原句子等功能。
sentenceSimilarity.py：通过tfidf模型计算相似度。
predict.py：主程序，用于读入数据，进行处理和计算并返回结果。

##怎么运行

###关于数据

本基线代码用了10000个seesion的会话数据，此数据是在京东公司客服和客户的真实聊天数据上做了脱敏处理后的数据。

###运行代码

1、进行数据处理，运行以下命令：

python dataProcessing.py

将数据集划分为train_set(questions、answers)和dev_set(devQuestions、devAnswers),此处是9:1的比例来划分的。并且以会话为单位，input格式为QAQAQ，output格式为A，Q表示question，A表示answer

2、进行训练和预测，运行命令：

python runModel.py

读入trian_questions、dev_questions，进行训练和预测。

##输出

输出文件格式如result.txt文件所示，即问题间的相似度以及每个样本的top15个候选答案。




