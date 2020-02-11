# train model
import sys
import time
import tensorflow as tf
import argparse
import os
import utils.config as cf
from model import BiLSTM_CRF
from data_process import random_embedding, read_dictionary, read_corpus,tag2label
from utils import train_utils
from tensorflow.contrib.crf import viterbi_decode
from utils.eval import conlleval

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
parser.add_argument('--update_embedding', type=train_utils.str2bool, default=params.update_embedding, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=params.embedding_dim, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=train_utils.str2bool, default=params.shuffle, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


# 参数部分
num_tags = len(tag2label)
word2id = read_dictionary(params.vocab_path)
summary_path = params.summary_path
model_path = params.store_path
result_path = params.result_path
embeddings = random_embedding(word2id, 300)
train_data = read_corpus(args.train_data)
test_data = read_corpus(args.test_data)
logger = cf.get_logger('logs/1.txt')

# 模型加载
model = BiLSTM_CRF(embeddings, args.update_embedding, args.hidden_dim, num_tags, args.clip, summary_path, args.optimizer)
model.build_graph()


def run_one_epoch(sess, train, dev, tag2label, epoch, saver):
    """
    训练模型，训练一个批次
    :param sess: 训练模型的一次会话
    :param train: 训练数据
    :param dev: 用来验证的数据
    :param tag2label: 标注转换为label的字典
    :param epoch: 批次的计数
    :param saver: 保存训练参数
    :return:
    """
    num_batches = (len(train) + args.batch_size -1) // args.batch_size
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    batches = train_utils.batch_yield(train, args.batch_size, word2id, tag2label)

    for step, (seqs, labels) in enumerate(batches):
        sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
        step_num = epoch * num_batches + step +1

        feed_dict, _ = train_utils.get_feed_dict(model, seqs, labels, args.lr, args.dropout)
        _, loss_train, summary, step_num_ = sess.run([model.train_op, model.loss, model.merged, model.global_step],
                                                     feed_dict=feed_dict)
        if step + 1 == 1 or (step + 1) % 20 == 0 or step + 1 == num_batches:
            print('logger info')
            logger.info('{} epoch {}, step {}, loss: {:.4}, total_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                   loss_train, step_num))

        if step + 1 == num_batches:
            saver.sace(sess, model_path, global_step=step_num)

    logger.info('=============test==============')
    label_list_dev, seq_len_list_dev = dev_one_epoch(sess, dev)
    evaluate(label_list_dev, seq_len_list_dev, dev, epoch)


def evaluate(label_list, seq_len_list, data, epoch=None):
    """

    :param label_list:
    :param seq_len_list:
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
        if  len(label_) != len(sent):
            print(sent)
            print(len(label_))
            print(tag)
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)
    epoch_num = str(epoch+1) if epoch is not None else 'test'
    label_path = os.path.join(result_path, 'label_' + epoch_num)
    metric_path = os.path.join(result_path, 'result_metric_' + epoch_num)
    for _ in conlleval(model_predict, label_path, metric_path):
        logger.info(_)


def dev_one_epoch(sess, dev):
    """

    :param sess: 训练的一次会话
    :param dev: 验证数据
    :return:
    """
    label_list, seq_len_list = [], []
    # 获取一个批次的句子中词的id以及标签
    for seqs, labels in train_utils.batch_yield(dev, args.batch_size, word2id, tag2label, shuffle=False):
        feed_dict, seq_len_list_ = train_utils.get_feed_dict(seqs, drop_keep=1.0)
        logits, transition_params = sess.run([model.logits, model.transition_params],
                                             feed_dict=feed_dict)
        label_list_ = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list_.append(viterbi_seq)

        label_list.extend(label_list_)
        seq_len_list.extend(seq_len_list_)
    return label_list, seq_len_list


def test(data, file):
    """
    模型测试
    :param data:测试数据
    :param file:模型
    """
    testSaver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        testSaver.restore(sess, file)
        label_list, seq_len_list = dev_one_epoch(sess, data)
        evaluate(label_list, seq_len_list, data)


def train(train_data, test_data):
    """
    进行模型训练
    :param train_data: 训练数据
    :param test_data: 测试数据
    :return: 
    """
    # model.train
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer()  # 初始化模型参数
        model.add_summary(sess)
        for epoch in range(args.epoch):
            run_one_epoch(sess, train_data, test_data, tag2label, epoch, saver)


def run(operation):
    """

    :param operation:
    :return:
    """
    if operation == 'train':
        train(train_data, test_data)

    if operation == 'test':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        test(test_data, ckpt_file)


if __name__ == '__main__':
    run(args.mode)