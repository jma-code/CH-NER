# train model
import random
import sys
import time
import tensorflow as tf

import argparse
import os
import numpy as np
from model import BiLSTM_CRF
from data_process import sentence2id, read_dictionary, read_corpus
import utils.config as cf

params = cf.ConfigTrain('train', 'config/params.conf')
params.load_config()

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default=params.trainData_path, help='train data source')
parser.add_argument('--test_data', type=str, default=params.testData_path, help='test data source')
parser.add_argument('--batch_size', type=int, default=params.batch_size, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=params.epoch, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=params.hidden_dim, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default=params.optimizer, help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=params.lr, help='learning rate')
parser.add_argument('--clip', type=float, default=params.clip, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=params.dropout, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=utils.str2bool, default=params.update_embedding, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=params.embedding_dim, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=utils.str2bool, default=params.shuffle, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


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

