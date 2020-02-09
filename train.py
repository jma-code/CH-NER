# train model
import random
import sys
import time
import tensorflow as tf
import os
import numpy as np
from model import BiLSTM_CRF
from data_process import sentence2id, read_dictionary, read_corpus


# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

# 参数部分
embedding_mat = np.random.uniform(-0.25, 0.25, (4756, 300))  # 4756*300
embedding_mat = np.float32(embedding_mat)
embeddings = embedding_mat
update_embedding = True
hidden_dim = 300
num_tags = 7  # len(tag2label)
clip_grad = 5.0
summary_path = "logs"
epoch_num = 4
batch_size = 20
word2id = read_dictionary("data/word2id.pkl")
lr = 0.001
drop_keep = 0.5
model_path = "checkpoints"
train_path = "data/train_data"
test_path = "data/test_data"


train_data = read_corpus(train_path)
test_data = read_corpus(test_path)
# 模型加载
model = BiLSTM_CRF(embeddings, update_embedding, hidden_dim, num_tags, clip_grad, summary_path)
model.build_graph()


def batch_yield(data, batch_size, vocab, tag2label):
    # 默认选择随机抽取
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


def pad_sequence(seqs, pad_mark):
    max_len = max(map(lambda x: len(x), seqs))
    seq_list, seq_len_list = [], []
    for seq in seqs:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq),0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq),max_len))
    return seq_list, seq_len_list


def get_feed_dict(seqs, labels, lr, drop_keep):
    word_ids, seq_len_list = pad_sequence(seqs, pad_mark=0)
    # feed_dict
    feed_dict = {"word_ids": word_ids, "sequence_lengths": seq_len_list}
    if labels is not None:
        labels_, _ = pad_sequence(labels, pad_mark = 0)
        feed_dict["labels"] = labels_
    if lr is not None:
        feed_dict["lr_pl"] = lr
    if drop_keep is not None:
        feed_dict["dropout_pl"] = drop_keep
        
    return feed_dict, seq_len_list


def dev_one_epoch(sess, dev):
    """

    :param sess:
    :param dev:
    :return:
    """
    # 需要predict方法
    pass


def run_one_epoch(sess, train, dev, tag2label, epoch, saver):
    num_batches = (len(train) + batch_size -1) // batch_size
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    batches = batch_yield(train, batch_size, word2id, tag2label)

    for step, (seqs, labels) in enumerate(batches):
        sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
        step_num = epoch * num_batches + step +1

        feed_dict, _ = get_feed_dict(seqs, labels, lr, drop_keep)
        _, loss_train, summary, step_num_ = sess.run([model.train_op, model.loss, model.merged, model.global_step], feed_dict=feed_dict)
        if step + 1 == 1 or (step + 1) % 20 == 0 or step + 1 ==num_batches:
            print('logger info')
            print("{} epoch {}, step {}, loss:{:.4}, total_step:{}".format(start_time, epoch + 1, step + 1, loss_train, step_num))

        # 写入日志文件
        # logging.info()

        if step + 1 == num_batches:
            saver.sace(sess, model_path, global_step=step_num)

    label_list_dev, seq_len_list_dev = dev_one_epoch(sess, dev)


# model.train
saver = tf.train.Saver(tf.global_variables())
with tf.Session(config=config) as sess:
    tf.global_variables_initializer()  # 初始化模型参数
    model.add_summary(sess)
    for epoch in range(epoch_num):
        run_one_epoch(sess, train_data, test_data, word2id, epoch, saver)

