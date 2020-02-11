import numpy as np
import os, time, sys, random
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

from model import BiLSTM_CRF
from data_process import sentence2id, read_dictionary
#import model
import utils


def predict_total(sess, sent, batch_size, vocab, tag2label, shuffle):
    batch_yield(sent, batch_size, vocab, tag2label, shuffle)
    get_tag = demo_one(sess, sent, batch_size, vocab, tag2label, shuffle)

def batch_yield(sent, batch_size, vocab, tag2label, shuffle):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(sent)

    seqs, labels = [], []
    for (sent_, tag_) in sent:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels



def demo_one(sess, sent, batch_size, vocab, tag2label, shuffle):
    """

    :param sess:
    :param sent:
    :return:
    """

    # batch_yield就是把输入的句子每个字的id返回，以及每个标签转化为对应的tag2label的值
    label_list = []
    for seqs, labels in batch_yield(sent, batch_size, vocab, tag2label, shuffle):
        label_list_, _ = BiLSTM_CRF.predict_one_batch(sess, seqs)
        label_list.extend(label_list_)
    label2tag = {}
    for tag, label in tag2label.items():
        label2tag[label] = tag if label != 0 else label
    tag = [label2tag[label] for label in label_list[0]]
    return tag

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


#在会话中启动图
sess = tf.Session()

input_sent = ['小', '明', '的', '大', '学', '在', '北', '京', '的', '北', '京', '大', '学']
get_sent = [(input_sent, ['O'] * len(input_sent))]
get_vocab = read_dictionary("data/word2id")
predict_total(sess, get_sent, 60, get_vocab, tag2label, False)


