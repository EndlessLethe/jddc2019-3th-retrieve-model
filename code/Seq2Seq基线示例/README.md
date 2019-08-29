## JD Dialog Challenge


## 关于该基线
#该基线是JD.com关于JD Dialog Challenge大赛向广大参赛选手们提供的开源基线代码，用到的模型为：Seq2Seq，该基线仅作参考。
#working_dir: (1).训练之后的模型保存文件  (2).词典文件
#data: (1).原训练语料, path：data/chat.txt  (2).数据清洗之后的训练语料：train.enc、train.dec、test.enc、test.dec (3).测试模型数据，测试问题test.txt和输出结果result.txt，path：data/test/
#seq2seq.ini: 参数配置文件
#dataProcessing.py: 数据清洗文件
#基线代码文件：data_utils.py、seq2seq_model.py、execute.py


## 关于数据
#本基线代码当前训练语料共计1万个seesion会话的数据（path：data/chat.txt），该数据是在京东公司客服和客户的真实聊天数据上做了脱敏处理后的数据。
#train.enc、train.dec、test.enc、test.dec作为模型的训练语料，都是基于文件chat.txt做的数据清洗，enc文件每行数据是同一会话中的QAQAQ，dec文件每行数据是同一会话中的A，Q表示用户回答，A表示客服/机器人回答，具体实现可参考数据处理文件dataProcessing.py


## requirement
# python3.5
# tensorflow1.0.0


## Train Model
# edit seq2seq.ini file to set mode = train
python execute.py


## Test
# edit seq2seq.ini file to set mode = test
#输入：输入文件格式(QAQAQ)，path(working_dir/test/test.txt)，注意正式比赛开始时会有100个问题，该基线文件只给出了50个问题
#输出：输出文件格式(A)，path(working_dir/test/result.txt)
python execute.py


## Notes
#本基线后续还会持续更新

