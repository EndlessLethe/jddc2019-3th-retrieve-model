# README

## How to run
python jddc_eval.py ./data/dev_question.txt ./out/test.txt

## How to run bert
```
--task_name=jd --vocab_file="./JDAI-BERT/vocab.txt" --bert_config_file="./JDAI-BERT/bert_config.json" --output_dir="./out" --do_eval=False --do_predict=True --data_dir=“./JDAI-BERT”
```

### build requirment
```
pipreqs --encoding=utf-8 ./
```

## score in Online Judge
1. 0.058596: 10per elmo 无停用词 k = 30 取中心答案
1. 0.056447: 1per elmo 无停用词 k = 30 取中心答案
1. 0.05353: 10per tfidf 无停用词 k = 20 取中心答案
2. 0.050728: 10per tfidf 无停用词 k = 15 取中心答案
2. 0.035051：100per tfidf 无停用词 k = 10 取中心答案
1. 0.021707: 10per tfidf 都用停用词 k = 15 取中心答案


## 目录结构：
- data
- code
- resource

Just put following dir in /code:
/ELMo
/DAM
/JDAI-BERT
/JDAI-WORD-EMBEDDIN1G

ELMo:https://github.com/HIT-SCIR/ELMoForManyLangs
DAM:https://github.com/baidu/Dialogue/tree/master/DAM
JDAI-BERT， JDAI-WORD-EMBEDDIN1G:https://github.com/jd-aig/nlp_baai
SMN_Pytorch：https://github.com/MaoGWLeon/SMN_Pytorch

## score in test set
在整个实验过程中，我们依次确定了一下参数：
1. 是否使用停用词 —— 不使用
2. 选择什么样的unsupervised reranker —— 中心
3. topk的k应该设置成多大 —— 30
4. lsi和lda的embedding效果 —— 被elmo完爆
5. 是否输入全部数据，或者只输入q —— 在所有模型上，只输入q都有提升
5. 是否在embedding时进行特殊字符的替换 —— 不替换
6. bert的提升效果 —— 


Note：
1. bert答案不稳定
2. elmo的答案也不稳定
3. 可能是因为batch的padding


Note：下面的表格没有按照上述顺序。



### bert
k=30 中心 替换 只输入q  
train size | embedding | 是否bert | score 
-|-|-|-
1per | elmo | ur | 
1per | elmo | bert | 
1per | tfidf | ur | 
1per | tfidf | bert | 
10per | tfidf | bert | 
10per | tfidf | ur | 

### 特殊字符
bert k=30 全部数据  
train size | embedding | 是否替换 | score 
-|-|-|-
0.01per | elmo | 不替换 | 0.010437204998100533
0.01per | tfidf | 不替换 | 0.012997054348273201
0.1per | elmo | 不替换 | 0.011414544946930654
1per | tfidf | 不替换 | 0.012848696840693473
0.01per | elmo | 替换 | 0.010099141591314545
0.01per | tfidf | 替换 | 0.010073682739222817
0.1per | elmo | 替换 | 0.011414544946930654
1per | tfidf | 替换 | 0.01059010512872386

结论：不替换特殊字符会带来提升

### 是否输入全部数据
使用不替换数据 k=30 中心  
train size | embedding | 是否全量数据 | score 
-|-|-|-
0.01per | tdidf | 只有q | 0.014383755741864323
0.01per | tdidf | 全部数据 | 0.012997054348273201
0.01per | elmo | 只有q | 0.015390332452465709
0.01per | elmo | 全部数据 | 0.010437204998100533
1per | tdidf | 只有q | 0.013483501888183026
1per | tdidf | 全部数据 | 0.012848696840693473


### unsupervised reranker
k = 30  
train size | embedding | 选取方式 | score 
-|-|-|-
1per | tfidf | 第一个 | 0.004963330927578965
10per | tfidf | 第一个 | 0.005813867175527974
1per | tfidf | 中心 | 0.013600463289462657
10per | tfidf | 中心 | 0.01278348452600705
0.01per | elmo | 第一个 | 0.006319689484710318
0.01per | elmo | 中心 | 0.013490131245278697

### 停用词调参
#### 总结
事实证明，tfidf不需要使用停用词。其他模型应该也不需要。

#### 1per
 train size | 停用词 | k | 选取方式 | score 
-|-|-|-
1per | 全不使用停用词 | 10 | 中心 | 0.008554900692186014
1per | TFIDF使用 | 10 | 中心 | 0.007695293483604558
1per | 检索时使用 | 10 | 中心 | 0.007417600955727696
1per | 聚类时使用 | 10 | 中心 | 0.008527690838414773
1per | 都使用 | 10 | 中心 | 0.006643151597518594

 train size | 停用词 | k | 选取方式 | score 
-|-|-|-|-
1per | 全不使用停用词 | 15 | 中心 | 0.010930076940092955
1per | TFIDF使用 | 15 | 中心 | 
1per | 检索时使用 | 15 | 中心 | 
1per | 聚类时使用 | 15 | 中心 | 
1per | 都使用 | 15 | 中心 | 

 train size | 停用词 | k | 选取方式 | score 
-|-|-|-|-
1per | 全不使用停用词 | 20 | 中心 | 0.01490869784672498
1per | TFIDF使用 | 20 | 中心 | 0.01346708503468178
1per | 检索时使用 | 20 | 中心 | 0.00850552305526961
1per | 聚类时使用 | 20 | 中心 | 0.007632402631464895
1per | 都使用 | 20 | 中心 | 0.006743577323141914

 train size | 停用词 | k | 选取方式 | score 
-|-|-|-|-
1per | 全不使用停用词 | 25 | 中心 | 0.011331763821217396
1per | TFIDF使用 | 25 | 中心 | 
1per | 检索时使用 | 25 | 中心 | 
1per | 聚类时使用 | 25 | 中心 | 
1per | 都使用 | 25 | 中心 | 

### k和停用词调参
#### 总结
tfidf确实不需要停用词。不过对于k来说，随着数据集的增大，有必要相应增大。

#### 10per
 train size | 停用词 | k | 选取方式 | score 
-|-|-|-|-
10per | 全不使用停用词 | 10 | 中心 | 0.008675446566447679
10per | 都使用 | 10 | 中心 | 
10per | 全不使用停用词 | 15 | 中心 | 0.010576244561356318
10per | TFIDF使用 | 15 | 中心 | 
10per | 检索时使用 | 15 | 中心 | 
10per | 聚类时使用 | 15 | 中心 | 
10per | 都使用 | 15 | 中心 | 
10per | 全不使用停用词 | 20 | 中心 | 0.010226410024601663
10per | 都使用 | 10 | 中心 |  
10per | 全不使用停用词 | 25 | 中心 | 0.012067605975253125
10per | 都使用 | 10 | 中心 |  
10per | 全不使用停用词 | 30 | 中心 | 0.01278348452600705 （虽然分数没有下降，但人眼可见的质量下降）
10per | 都使用 | 10 | 中心 |  
10per | 全不使用停用词 | 50 | 中心 | 0.01442813116502754 （开始出现大量通用性高的重复回答）
10per | 都使用 | 10 | 中心 |  

### 主题模型topic_num调参
#### lsi调参
k = 30  
 train size | model | topic_num | score 
-|-|-|-
1per | lsi | 20 | 0.01071816672917079
1per | lsi | 30 | 0.01104208774490005
1per | lsi | 40 | 0.013137852396458436
1per | lsi | 60 | 0.012325831746581317
1per | lsi | 80 | 0.011505039619989454
1per | lsi | 120 | 0.011522613952786432
10per | lsi | 20 | 0.010151781262281053
10per | lsi | 40 | 0.008701707439616796
10per | lsi | 60 | 0.011182269704425877

k = 20  
 train size | model | topic_num | score 
-|-|-|-
10per | lsi | 40 | 0.008171042618110087
10per | lsi | 60 | 0.010761877752782888


#### lda调参
k = 20  
 train size | model | topic_num | score 
-|-|-|-
10per | lda | 40 | 0.008900965387194669
10per | lda | 60 | 0.007588860032794996


k = 30  
 train size | model | topic_num | score 
-|-|-|-
10per | lda | 40 | 0.008900965387194669
10per | lda | 60 | 0.01256071632922341
10per | lda | 80 | 0.006515710589744413
