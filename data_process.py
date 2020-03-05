import os
import pickle
import numpy as np
import utils.config as cf


def total(corpus_path, vocab_path, embedding_dim):
    """
    Created by jty
    预处理模块总函数。
    输入：训练数据路径、word2id字典保存路径、词频阈值、测试输入句子、向量化维数
    输出：训练数据向量化结果、测试输入句子id
    """
    read_corpus(corpus_path)
    vocab_build(vocab_path, corpus_path)
    get_word2id = read_dictionary(vocab_path)
    get_embedding_mat = random_embedding(get_word2id, embedding_dim)
    return get_embedding_mat

def data_clean(corpus_path, clean_path):
    """
    Created by jty
    输入train_data文件的路径，读取训练集的语料，输出train_data
    read corpus and return the list of samples
    :param：corpus_path
    :return: data
    """
    with open(corpus_path, encoding='utf-8') as fw:
        with open(clean_path, 'w', encoding='utf-8') as fwn:
            lines = fw.readlines()
            for line in lines:
                if line != '\n':
                    char, label = line.strip().split()
                    if label == 'B-PER.NAM' or label == 'B-PER.NOM':
                        label = 'B-PER'
                    if label == 'I-PER.NAM' or label == 'I-PER.NOM':
                        label = 'I-PER'
                    if label == 'B-ORG.NAM' or label == 'B-ORG.NOM':
                        label = 'B-ORG'
                    if label == 'I-ORG.NAM' or label == 'I-ORG.NOM':
                        label = 'I-ORG'
                    if label == 'B-LOC.NAM' or label == 'B-LOC.NOM' or label == 'B-GPE.NAM':
                        label = 'B-LOC'
                    if label == 'I-LOC.NAM' or label == 'I-LOC.NOM' or label == 'I-GPE.NAM':
                        label = 'I-LOC'
                    fwn.write(char + ' ' + label + '\n')
                else:
                    fwn.write("\n")
    '''
    per1 = 0
    per2 = 0
    loc1 = 0
    loc2 = 0
    org1 = 0
    org2 = 0
    org3 = 0
    sentence_count = 0
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        if line.find("B-PER.NOM") > -1:
            per1 = per1 + 1
        if line.find("B-PER.NAM") > -1:
            per2 = per2 + 1
        if line.find("B-LOC.NOM") > -1:
            loc1 = loc1 + 1
        if line.find("B-LOC.NAM") > -1:
            loc2 = loc2 + 1
        if line.find("B-GPE.NAM") > -1:
            org1 = org1 + 1
        if line.find("B-ORG.NOM") > -1:
            org2 = org2 + 1
        if line.find("B-ORG.NAM") > -1:
            org3 = org3 + 1
        if line == '\n':
            sentence_count = sentence_count + 1
    return per1, per2, loc1, loc2, org1, org2, org3, sentence_count
    '''


def read_corpus(corpus_path):
    """
    Created by jty
    输入train_data文件的路径，读取训练集的语料，输出train_data
    read corpus and return the list of samples
    :param：corpus_path
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        '''lines的形状为['北\tB-LOC\n','京\tI-LOC\n','的\tO\n','...']总共有2220537个字及对应的tag'''
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            # char 与 label之间有个空格
            # line.strip()的意思是去掉每句话句首句尾的空格
            # .split()的意思是根据空格来把整句话切割成一片片独立的字符串放到数组中，同时删除句子中的换行符号\n
            [char, label] = line.strip().split()
            # 把一个个的字放进sent_
            sent_.append(char)
            # 把字后面的tag放进tag_
            tag_.append(label)
            # 一句话结束了，添加到data
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    """ data的形状为[(['我',在'北','京'],['O','O','B-LOC','I-LOC'])...第一句话
        (['我',在'天','安','门'],['O','O','B-LOC','I-LOC','I-LOC'])...第二句话  
        ( 第三句话 )  ] 总共有50658句话
    """
    return data


def vocab_build(vocab_path, corpus_path):
    """
    Created by jty
    由train_data来构造一个(统计非重复字)字典{'第一个字':[对应的id,该字出现的次数],'第二个字':[对应的id,该字出现的次数], , ,}
    去除低频词，生成一个word_id的字典并保存在输入的vocab_path的路径下，保存的方法是pickle模块自带的dump方法，保存后的文件格式
    是word2id.pkl文件
    :param vocab_path:
    :param corpus_path:
    :return: word2id
    """
    data = read_corpus(corpus_path)
    word2id = {}
    # word2id = data
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():  # 检查是否为0-9
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):  # 检查是否为英文字母，ASCII码判别
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    # print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        # 序列化到名字为word2id。pkl文件
        pickle.dump(word2id, fw)

    return word2id


def read_dictionary(vocab_path):
    """
    Created by jty
    通过pickle模块自带的load方法(反序列化方法)加载输出word2id
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    # print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """
    Created by jty
    输入vocab，vocab就是前面得到的word2id，embedding_dim=300
    :param vocab:
    :param embedding_dim:
    :return:
    """
    # 返回一个len(vocab)*embedding_dim=3905*300的矩阵(每个字投射到300维)作为初始值
    # numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    # 参数介绍:
    #     
    #     low: 采样下界，float类型，默认值为0；
    #     high: 采样上界，float类型，默认值为1；
    #     size: 输出样本数目，为int或元组(tuple)
    # 类型，例如，size = (m, n, k), 则输出m * n * k个样本，缺省时输出1个值。
    #
    # 返回值：ndarray类型，其形状和参数size中描述一致。
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


#data_clean('data/weibo_dev.txt', 'data/weibo_clean_dev.txt')
if __name__ == '__main__':
    params = cf.ConfigProcess('process', 'config/params.conf')
    params.load_config()
    get_embedding_mat = total(params.corpus_path, params.vocab_path, params.embedding_dim)
