import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from copy import deepcopy
import re

from model import BiLSTM_CRF
from utils import train_utils
from data_process import read_dictionary
import utils.config as cf

# 参数部分
params = cf.ConfigPredict('predict', 'config/params.conf')
params.load_config()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3


def predict_one_batch(model, ses, seqs):
    """
    Created by jty
    预测引擎，输入句子id和保存好的模型参数进行预测，输出标签id
    :param ses: 使用会话
    :param seqs: 句子id
    :return: label_list seq_len_list 标签id 句子id
    """
    feed_dict, seq_len_list = train_utils.get_feed_dict(model, seqs, drop_keep=1.0)

    # transition_params代表转移概率，由crf_log_likelihood方法计算出
    log_its, transition_params = ses.run([model.log_its, model.transition_params],
                                         feed_dict=feed_dict)
    label_list = []
    # 默认使用CRF
    for log_it, seq_len in zip(log_its, seq_len_list):
        vtb_seq, _ = viterbi_decode(log_it[:seq_len], transition_params)
        label_list.append(vtb_seq)
    return label_list, seq_len_list


def demo_one(model, ses, sent, batch_size, vocab, shuffle, tag2label):
    """
    Created by jty
    输入句子，得到预测标签id，并转化为label
    :param model: 保存好的模型
    :param ses: 使用会话
    :param sent: 输入要进行实体抽取的句子
    :param batch_size: 每次预测的句子数
    :param vocab:  word2id
    :param shuffle: 默认为False
    :return: tag 预测标签
    """

    # batch_yield就是把输入的句子每个字的id返回，以及每个标签转化为对应的tag2label的值
    label_list = []
    for seqs, labels in train_utils.batch_yield(sent, batch_size, vocab, tag2label, shuffle):
        label_list_, _ = predict_one_batch(model, ses, seqs)
        label_list.extend(label_list_)
    label2tag = {}
    for tag, label in tag2label.items():
        label2tag[label] = tag if label != 0 else label
    tag = [label2tag[label] for label in label_list[0]]
    return tag


"""
Created by jty
数据后处理
根据输入的tag和句子返回对应的字符
其中包括抽取出对应的人名、地名、组织名
"""


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG


# 输出PER对应的字符
def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i + 1 == length:
                per = per.strip()
                per.replace('\n', '')
                per.replace('\r', '')
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i + 1 == length:
                per = per.strip()
                per.replace('\n', '')
                per.replace('\r', '')
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                per=per.strip()
                per.replace('\n', '')
                per.replace('\r', '')
                PER.append(per)
                del per
            continue
    return PER


# 输出LOC对应的字符
def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                loc = loc.strip()
                loc.replace('\n', '')
                loc.replace('\r', '')
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i + 1 == length:
                loc = loc.strip()
                loc.replace('\n', '')
                loc.replace('\r', '')
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                loc = loc.strip()
                loc.replace('\n', '')
                loc.replace('\r', '')
                LOC.append(loc)
                del loc
            continue
    return LOC


# 输出ORG对应的字符
def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i + 1 == length:
                org = org.strip()
                org = org.replace('\n', '')
                org = org.replace('\r', '')
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i + 1 == length:
                org = org.strip()
                org = org.replace('\n', '')
                org = org.replace('\r', '')
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                org = org.strip()
                org = org.replace('\n', '')
                org = org.replace('\r', '')
                ORG.append(org)
                del org
            continue
    return ORG


def predict(model, batch_size, vocab, tag2label, demo_sent, shuffle=False):
    """
    Created by jty
    预测模块总函数。
    输入：保存好的模型、每次预测的句子数、word2id字典、交互界面输入的需要实体抽取的句子
    输出：实体抽取的结果
    :param model: 保存好的模型
    :param batch_size: 每次预测的句子数
    :param vocab: word2id
    :param shuffle: 默认为False
    """
    s_id = 1
    sent_id = {}
    ckpt_file = tf.train.latest_checkpoint(params.model_path)
    print(ckpt_file)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        # print('============= demo =============')
        saver.restore(sess, ckpt_file)
        # print('Please input your sentence:')
        # demo_sent = input()
        #demo_sent = '我在北京上北京大学'
        if demo_sent == '' or demo_sent.isspace():
            print('See you next time!')
        else:
            # 打上id标签
            for word in demo_sent:
                sent_id[s_id] = word
                s_id += 1
            demo_sent = list(demo_sent.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tag = demo_one(model, sess, demo_data, batch_size, vocab, shuffle, tag2label)
            PER, LOC, ORG = get_entity(tag, demo_sent)
            PER_local = {}
            LOC_local = {}
            ORG_local = {}
            p_id = 1
            l_id = 1
            o_id = 1
            PER_mess = {}
            LOC_mess = {}
            ORG_mess = {}
            # 抽取PER实体长度、位置信息
            i = 1
            for word in PER:
                PER_local['item'] = word
                PER_local['tag'] = 'PER'
                PER_local['length'] = len(word)
                for j in range(i, len(sent_id)):
                    if word[0] == sent_id[j]:
                        PER_local['offset'] = j
                        i = j + len(word)
                        break
                PER_mess[p_id] = deepcopy(PER_local)
                p_id += 1
            # 抽取LOC实体长度、位置信息
            i = 1
            for word in LOC:
                LOC_local['item'] = word
                LOC_local['tag'] = 'LOC'
                LOC_local['length'] = len(word)
                for j in range(i, len(sent_id)):
                    if word[0] == sent_id[j]:
                        LOC_local['offset'] = j
                        i = j + len(word)
                        break
                LOC_mess[l_id] = deepcopy(LOC_local)
                l_id += 1
            # 抽取ORG实体长度、位置信息
            i = 1
            for word in ORG:
                ORG_local['item'] = word
                ORG_local['tag'] = 'ORG'
                ORG_local['length'] = len(word)
                for j in range(i, len(sent_id)):
                    if word[0] == sent_id[j]:
                        ORG_local['offset'] = j
                        i = j + len(word)
                        break
                ORG_mess[o_id] = deepcopy(ORG_local)
                o_id += 1
            #print(PER_mess, LOC_mess, ORG_mess)
            return PER_mess, LOC_mess, ORG_mess
def run(demo_sent, flag=False):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(read_dictionary(params.vocab_path)), params.embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    embeddings = embedding_mat
    num_tags = len(params.tag2label)
    summary_path = "logs"
    model = BiLSTM_CRF(embeddings, params.update_embedding, params.hidden_dim, num_tags, params.clip, summary_path,
                       params.optimizer)
    model.build_graph()
    PER_mess, LOC_mess, ORG_mess = predict(model, params.batch_size, read_dictionary(params.vocab_path), params.tag2label, demo_sent)
    if flag:
        return PER_mess, LOC_mess, ORG_mess

#run('我在北京上北京大学,周恩来是中国总理,我喜欢北京。我在清华大学，毛泽东是中国主席，他去过苏联。')