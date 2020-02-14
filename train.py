import sys
import time
import tensorflow as tf
import argparse
import os
import utils.config as cf
from model import BiLSTM_CRF
from data_process import random_embedding, read_dictionary, read_corpus, tag2label
from utils import train_utils
from tensorflow.contrib.crf import viterbi_decode
from utils.eval import conlleval

class Train(object):
    params = ''

    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def evaluate(self):
        pass

params = cf.ConfigTrain('train', 'config/params.conf')
params.load_config()

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default=params.trainData_path, help='train data source')
parser.add_argument('--test_data', type=str, default=params.testData_path, help='test data source')
parser.add_argument('--batch_size', type=int, default=params.batch_size, help='#sample of each minbatch')
parser.add_argument('--epoch', type=int, default=params.epoch, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=params.hidden_dim, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default=params.optimizer,
                    help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=params.lr, help='learning rate')
parser.add_argument('--clip', type=float, default=params.clip, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=params.dropout, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=train_utils.str2bool, default=params.update_embedding,
                    help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                    help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=params.embedding_dim, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=train_utils.str2bool, default=params.shuffle,
                    help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()

# 参数部分
word2id = read_dictionary(params.vocab_path)
embeddings = random_embedding(word2id, args.embedding_dim)
logger = cf.get_logger('logs/1.txt')


def run_one_epoch(model, sess, train_corpus, dev, tag_label, epoch, saver):
    """
    训练模型，训练一个批次
    :param model: 模型
    :param sess: 训练模型的一次会话
    :param train_corpus: 训练数据
    :param dev: 用来验证的数据
    :param tag_label: 标注转换为label的字典
    :param epoch: 批次的计数
    :param saver: 保存训练参数
    :return:
    """
    num_batches = (len(train_corpus) + args.batch_size - 1) // args.batch_size
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    batches = train_utils.batch_yield(train_corpus, args.batch_size, word2id, tag_label)

    for step, (seqs, labels) in enumerate(batches):
        sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
        step_num = epoch * num_batches + step + 1

        feed_dict, _ = train_utils.get_feed_dict(model, seqs, labels, args.lr, args.dropout)

        _, loss_train, summary, step_num_ = sess.run([model.train_op, model.loss, model.merged, model.global_step],
                                                     feed_dict=feed_dict)

        if step + 1 == 1 or (step + 1) % args.batch_size == 0 or step + 1 == num_batches:
            print('logger info')
            logger.info('{} epoch {}, step {}, loss: {:.4}, total_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                   loss_train, step_num))

        if step + 1 == num_batches:
            saver.save(sess, params.store_path, global_step=step_num)

    logger.info('=============test==============')
    label_list_dev, seq_len_list_dev = dev_one_epoch(model, sess, dev)
    evaluate(label_list_dev, dev, epoch)


def evaluate(label_list, data, epoch=None):
    """
    评估模型标注结果
    :param label_list:
    :param data:
    :param epoch:
    :return:
    """
    label2tag = {}
    for tag, label in tag2label.items():
        label2tag[label] = tag if label != 0 else label

    model_predict = []
    for label_, (sent, tag) in zip(label_list, data):
        tag_ = [label2tag[label__] for label__ in label_]
        sent_res = []
        if len(label_) != len(sent):
            print(sent)
            print(len(label_))
            print(tag)
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)
    epoch_num = str(epoch + 1) if epoch is not None else 'test'
    label_path = os.path.join(params.result_path, 'label_' + epoch_num)
    metric_path = os.path.join(params.result_path, 'result_metric_' + epoch_num)
    for _ in conlleval(model_predict, label_path, metric_path):
        logger.info(_)


def dev_one_epoch(model, sess, dev):
    """
    对一个epoch进行验证
    :param model: 运行的模型
    :param sess: 训练的一次会话
    :param dev: 验证数据
    :return:
    """
    label_list, seq_len_list = [], []
    # 获取一个批次的句子中词的id以及标签
    for seqs, labels in train_utils.batch_yield(dev, args.batch_size, word2id, tag2label, shuffle=False):
        feed_dict, seq_len_list_ = train_utils.get_feed_dict(model, seqs, drop_keep=1.0)
        log_its, transition_params = sess.run([model.log_its, model.transition_params],
                                              feed_dict=feed_dict)
        label_list_ = []
        for log_it, seq_len in zip(log_its, seq_len_list_):
            vtb_seq, _ = viterbi_decode(log_it[:seq_len], transition_params)
            label_list_.append(vtb_seq)

        label_list.extend(label_list_)
        seq_len_list.extend(seq_len_list_)
    return label_list, seq_len_list


def test(data, file):
    """
    模型测试
    :param data:测试数据
    :param file:模型
    """
    model = BiLSTM_CRF(embeddings, args.update_embedding, args.hidden_dim, len(tag2label), args.clip,
                       params.summary_path, args.optimizer)
    model.build_graph()
    testsaver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        testsaver.restore(sess, file)
        label_list, seq_len_list = dev_one_epoch(model, sess, data)
        evaluate(label_list, data)


def train(train_corpus, test_corpus):
    """
    进行模型训练
    :param train_corpus: 训练数据
    :param test_corpus: 测试数据
    :return: 
    """
    # model.train
    model = BiLSTM_CRF(embeddings, args.update_embedding, args.hidden_dim, len(tag2label), args.clip,
                       params.summary_path, args.optimizer)
    model.build_graph()

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=config) as sess:
        # tf.global_variables_initializer()  # 初始化模型参数
        sess.run(model.init_op)
        model.add_summary(sess)

        for epoch in range(args.epoch):
            run_one_epoch(model, sess, train_corpus, test_corpus, tag2label, epoch, saver)


def run(operation):
    """
    选择对模型的操作，包括训练和测试
    :param operation:
    :return:
    """
    if operation == 'train':
        train_data = read_corpus(args.train_data)
        test_data = read_corpus(args.test_data)
        train(train_data, test_data)

    if operation == 'test':
        chk_file = tf.train.latest_checkpoint(params.result_path)
        test_data = read_corpus(args.test_data)
        test(test_data, chk_file)


if __name__ == '__main__':
    run(args.mode)
