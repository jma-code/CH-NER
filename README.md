# 使用BiLSTM+CRF实现中文命名实体识别
## 环境要求：
 - Python 3.6
 - TensorFlow >= 1.14.1
## 模型
该模型与文献[1]和[2]提供的模型相似。其结构如下图所示：
![Image text](https://github.com/jma-code/NER/blob/master/image_store/network.png)

对于一个中文句子，这个句子中的每个字符都有一个属于集合{O，B-PER，I-PER，B-LOC，I-LOC，B-ORG，I-ORG}的标记。
第一层，look-up层，旨在将每个字符表示从一个独热向量转换为字符嵌入。
第二层，BiLSTM层，可以有效地利用过去和将来的输入信息，自动提取特征。
第三层，CRF层，在一个句子中为每个字符标记标签。如果我们使用Softmax进行标记，我们可能会得到非随机标记序列，因为Softmax层独立地标记每个位置。我们知道“I-LOC”不能跟在“B-PER”后面，但Softmax不知道。与Softmax相比，CRF层可以利用句子级的标签信息，对两个不同标签的转换行为进行建模。

## 数据集
数据集使用的是微软亚研院提供的的中文数据集
![Image text](https://github.com/jma-code/NER/blob/master/image_store/corpus.png)

### 数据文件
目录.data/下包括
+ 预处理好的数据，train_data,test_data
+ 一个词汇表文件word2id，它将每个字符映射到一个唯一的id
要生成词汇表文件，请参考data_process.py中的代码。

### 数据形式
每个数据文件应采用以下格式：
![Image text](https://github.com/jma-code/NER/blob/master/image_store/datadescrib.png)

如果想要使用自己的数据集，你需要：
+ 把你的语料库转换成上面的格式
+ 重新生成一个词汇表文件

## 运行

### train
python3 main.py --mode=train

### test
python3 main.py --mode=test

### predict
python3 main.py --mode=predict

## 相关文献
+ [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991v1.pdf)
+ [Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030/)
+ [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_20)
+ [https://github.com/guillaumegenthial/sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging)
