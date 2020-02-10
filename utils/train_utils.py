# 添加在训练时会使用的方法工具
import argparse
import random

from data_process import sentence2id


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
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

    :param sequences:
    :param pad_mark:
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


def get_feed_dict(seqs, labels=None, lr=None, drop_keep=None):
    word_ids, seq_len_list = pad_sequence(seqs, pad_mark=0)
    # feed_dict
    feed_dict = {"word_ids": word_ids, "sequence_lengths": seq_len_list}
    if labels is not None:
        labels_, _ = pad_sequence(labels, pad_mark=0)
        feed_dict["labels"] = labels_
    if lr is not None:
        feed_dict["lr_pl"] = lr
    if drop_keep is not None:
        feed_dict["dropout_pl"] = drop_keep

    return feed_dict, seq_len_list
