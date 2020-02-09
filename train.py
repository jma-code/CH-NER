# train model

import argparse
import os
import numpy as np
import tensorflow as tf

from model import BiLSTM_CRF
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


embedding_mat = np.random.uniform(-0.25, 0.25, (4756, 300))  # 4756*300
embedding_mat = np.float32(embedding_mat)
embeddings = embedding_mat
update_embedding = True
hidden_dim = 300
num_tags = 7  # len(tag2label)
clip_grad = 5.0
summary_path = "logs"
model = BiLSTM_CRF(embeddings, update_embedding, hidden_dim, num_tags, clip_grad, summary_path)
model.build_graph()

# model.train
# saver = tf.train.Saver(tf.global_variables())
with tf.Session(config=config) as sess:
    tf.global_variables_initializer()  # 初始化模型参数
    model.add_summary(sess)
