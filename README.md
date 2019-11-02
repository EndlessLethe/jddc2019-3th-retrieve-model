# README

## 前言
很高兴我和小伙伴“网数ICT小分队”在JDDC 2019获得并列亚军（第三名）。 这是由我负责的检索模型部分。  

现在回过来看这个检索模型，在学习了决赛中各个队伍的优秀解决方案后，我觉得还有很多可以改进的地方，也有前沿的方法可以引入。  
但是随着比赛的结束，我可能也没有时间继续完善和实验更多的方法了，只能在这里提供一个粗糙的模型和框架，供大家参考。


## 目录结构：
- data: 存放运行模型所需要的data  
- code:   
|- data_processer.py: 训练数据和中间数据、最后结构的生成  
|- embedding_model_loader.py: 词/字embedding模型的简单工厂，提供了训练和使用模型的基本函数  
|- jieba_seg.py: 结巴分词的包装类，添加了停用词  
|- result_evaluator.py: 和线上评测相同的BLEU4，并且使用了平滑函数  
|- run_model.py: 整个模型的pipeline类，将所有模块连接起来  
|- session_processer.py: 读取和生成评测所需要的格式  
|- unsupervised_reranker.py：在rerank阶段的无监督模型（bert通过重写的run_classifier调用）  
|- dataeda.ipynb: 做数据分析和数据预处理  
- resource: 部分工作的截图和记录  
- modified code for models： 存放了修改过后的bert/elmo代码

## 外部模型下载
模型调用的外部模型需要在对应github仓库下载，并放置到对应目录中。
在这个仓库的modified code for models文件夹中，包括了我部分修改过的bert/elmo代码。如果需要运行模型，需要复制并覆盖外部模型的原文件。

Just put following dir in /code:
/ELMo
/DAM
/bert
/bert/JDAI-BERT
/bert/JDAI-WORD-EMBEDDIN1G

ELMo:https://github.com/HIT-SCIR/ELMoForManyLangs  
DAM:https://github.com/baidu/Dialogue/tree/master/DAM  
bert:https://github.com/google-research/bert  
JDAI-BERT， JDAI-WORD-EMBEDDIN1G:https://github.com/jd-aig/nlp_baai  
SMN_Pytorch：https://github.com/MaoGWLeon/SMN_Pytorch

Note: Please create a dir "out" in "/bert" for outputing

## 数据EDA以及预处理
### EDA
1. 句子长短分布
2. 对话轮数分布
3. 异常数据再分析
3. 常见对话及其所占比例

### 预处理
1. 将不同订单号和链接用同一特殊字段代替，去掉表情
2. 将连续的多个Q、多个A的句子合并成一句
3. 去掉空白的、只有符号或表情的句子
4. 去掉长度超过阈值的句子

## 检索式模型
The Dialog Systems Technology Challenges 7 (DSTC7)定义了5个子问题：

No | Subtask Description
-|-
1 | Select the next response from a set of 100 choices that contains 1 correct response
2 | Select the next response from a set of 120,000 choices
3 | Select the next response(s) from a set of 100 choices that contains between 1 and 5 correct responses
4| Select the next response or NONE from a set of 100 choices that contains 0 or 1 correct response
5| Select the next response from a set of 100 choices, given access to an external dataset

对于本次大赛的检索式的对话模型，我们重点关注问题1和2——如何从大量对话数据中选取top k的少量候选数据？以及如何使用更加精准的重排（rerank）模型，从候选数据中选取最为匹配的答案。
因此我们的检索式模型分为两大部分——粗筛模块和重排模块。
而因为模型面临的场景是多轮会话，对于每个模块需要考虑是否引入历史信息。

### 粗筛模块
1. 根据用户的问题与语料库中的问题，计算语句基于词或字的embedding
2. 将词或字的embedding通过加权得到句子的embedding
3. 将问题的sentence embedding和语料库中的，计算余弦相似度
4. 选取top k的相似问题，将k个问题和答案都传入重排模块

#### 是否使用对话历史
经过反复验证，结论是不使用：
1. 句向量的表示能力有局限性。随着句子长度的增加，未必能表示出整个句子的关键信息
2. 在当前问题的基础上附加过长的历史信息会导致词向量中当前问题的信息保留较少，检索时重心偏于历史信息的表示，而忽略了当前问题
3. 无法解决对话中“话题漂移”的现象，即之前的历史并不能代表当前问题的语境。此时的检索相当于引入了错误的信息。

#### 粗筛embedding
我们尝试了以下各种embedding：
1. bow
2. tfidf
3. lsi
4. lda
5. 电商领域基于skip-gram的字embedding
6. 基于上下文的elmo词embedding
7. 基于上下文的bert字embedding

优缺点概述：
1. tfidf是一个有稳定表现的无参数向量化表示方法，有着较好的表示效果。
2. lsi和lda的表示效果受到主题数量选取的影响很大，而且随着数据量的指数增大，表示效果不能提升。
3. 基于skip-gram的字embedding，表示效果优于tfidf。虽然没有包含上下文信息，但是考虑到模型的应用场景为电商领域，语言的多义性会较少，而且运行速度快，计算字向量时查表即可。
4. elmo得益于其对上下文语义的理解，生成的词向量能够包含上下文的语义，效果好于tfidf和skip-gram。但是缺点也同样是因为对于不同上下文的同一个词都有不同的向量表示，随着语料集的指数扩大，语料集生成词向量的过程需要非常耗时。
5. bert的优缺点和elmo类似，虽然能够建模上下文信息，但是所有向量需要长时间生成，大量空间储存。
综上所述，我们最后选择了电商领域基于skip-gram的字embedding。

### 重排模块
1. 将候选的top k个答案传入重拍模型
2. 重拍模型输出一个最优的答案

#### 是否使用对话历史
使用：
通过引入神经网络模型，模型对对话历史的建模能力大大提升，从而历史中的信息可以有效和正确地被利用。

#### 重排模型的选取
我们尝试了以下重排模型：
1. 中心选取
3. SMN
2. bert

Note：因为SMN代码的版本和数据格式和当前模型差异极大，在本模型中没有接入SMN，只在本地运行和评测了SMN模型

优缺点概述：
1. 中心选取的方法是假设top k答案大多数都是比较匹配的回答，在top k中选取和所有答案相似度最高的、共性最强的
2. SMN虽然能够对历史信息进行建模，但是模型设计以现在的眼光来看比较粗糙，提取能力不强
3. bert作为当下最流行的模型，通过海量预训练数据，有着强大的语言理解能力。通过全量对话数据的fine tuning，能够有效结合历史信息和当前问题。


## score in Online Judge
1. 0.0618: 10per bert 无停用词 k = 30 中心  只输入q 不替换
1. 0.058596: 10per elmo 无停用词 k = 30 中心
2. 0.058389: 10per tfidf 无停用词 k = 30 中心
1. 0.056447: 1per elmo 无停用词 k = 30 中心
1. 0.05353: 10per tfidf 无停用词 k = 20 中心
2. 0.050728: 10per tfidf 无停用词 k = 15 中心
2. 0.035051：100per tfidf 无停用词 k = 10 中心
1. 0.026642：0.1per elmo k = 30 bert
1. 0.021707: 10per tfidf 都用停用词 k = 15 中心

Note: 对于线上提交，因为bert模型太大，浏览器会崩溃，所以没有提交。只在人工评测的复赛阶段提交。

