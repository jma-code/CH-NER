import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

from model import BiLSTM_CRF
from utils import train_utils
from data_process import tag2label
import utils.config as cf
import data_process

# 参数部分
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
params = cf.ConfigPredict('predict', 'config/params.conf')
params.load_config()
embedding_mat = np.random.uniform(-0.25, 0.25, (4756, 300))  # 4756*300
embedding_mat = np.float32(embedding_mat)
embeddings = embedding_mat
num_tags = len(data_process.tag2label)
summary_path = "logs"

def predict_one_batch(sess, seqs):
    """

    :param sess:
    :param seqs:
    :return: label_list
                 seq_len_list
    """
    feed_dict, seq_len_list = train_utils.get_feed_dict(seqs, drop_keep=1.0)

    # transition_params代表转移概率，由crf_log_likelihood方法计算出
    logits, transition_params = sess.run([model.logits, model.transition_params],
                                         feed_dict=feed_dict)
    label_list = []
    # 默认使用CRF
    for logit, seq_len in zip(logits, seq_len_list):
        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
        label_list.append(viterbi_seq)
    return label_list, seq_len_list

def demo_one(sess, sent, batch_size, vocab, tag2label, shuffle):
    """

    :param sess:
    :param sent:
    :return:
    """

    # batch_yield就是把输入的句子每个字的id返回，以及每个标签转化为对应的tag2label的值
    label_list = []
    for seqs, labels in train_utils.batch_yield(sent, batch_size, vocab, tag2label, shuffle):
        label_list_, _ = predict_one_batch(sess, seqs)
        label_list.extend(label_list_)
    label2tag = {}
    for tag, label in tag2label.items():
        label2tag[label] = tag if label != 0 else label
    tag = [label2tag[label] for label in label_list[0]]
    return tag

model = BiLSTM_CRF(embeddings, params.update_embedding, params.hidden_dim, num_tags, params.clip, summary_path, params.optimizer)
model.build_graph()
input_sent = ['小', '明', '的', '大', '学', '在', '北', '京', '的', '北', '京', '大', '学']
get_sent = [(input_sent, ['O'] * len(input_sent))]
get_vocab = data_process.read_dictionary("data/word2id")
#在会话中启动图
with tf.Session(config=config) as sess:
    demo_one(sess, get_sent, 60, get_vocab, tag2label, False)


