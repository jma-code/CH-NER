# 添加在训练时会使用的方法工具
import argparse
import random


def sentence2id(sent, word2id):
    """

    :param sent: 句子
    :param word2id: 字典
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """
    将输入的数据转换为模型可以训练的输入数据
    :param data: 输入数据
    :param batch_size: 一次处理的数据
    :param vocab: 词典，word2id
    :param tag2label: 标注转换为label
    :param shuffle: 是否随机采样
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs, labels


def pad_sequence(sequences, pad_mark=0):
    """
    对输入序列进行填充，使用pad_mark进行填充
    :param sequences: 输入的序列
    :param pad_mark: 填充的字符
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)  # 后边填充0到300维
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list  # 保留填充后向量和填充前向量


def get_feed_dict(model, seqs, labels=None, lr=None, drop_keep=None):
    """
    将batch_yield的数据进行填充和模型训练参数一起传入模型
    :param model: (传入正在训练的模型)
    :param seqs: batch_yield后的序列
    :param labels: 携带的标签
    :param lr: 学习率
    :param drop_keep: 训练中参数随机放弃的百分比
    :return:
    """
    word_ids, seq_len_list = pad_sequence(seqs, pad_mark=0)
    # feed_dict
    feed_dict = {model.word_ids: word_ids,
                 model.sequence_lengths: seq_len_list}
    if labels is not None:
        labels_, _ = pad_sequence(labels, pad_mark=0)
        feed_dict[model.labels] = labels_
    if lr is not None:
        feed_dict[model.lr_pl] = lr
    if drop_keep is not None:
        feed_dict[model.dropout_pl] = drop_keep

    return feed_dict, seq_len_list
