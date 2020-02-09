# train model

import tensorflow as tf
import os
import numpy as np
from model import BiLSTM_CRF


# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

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
