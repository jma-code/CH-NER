# 创建模型类，规范化命名空间。
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.rnn import LSTMCell


class BiLSTM_CRF(object):
    # 初始化类中需要的变量(到时候考虑config)
    def __init__(self, embeddings, update_embedding, hidden_dim, num_tags, clip_grad, log_path, optimizer):
        self.embeddings = embeddings
        self.update_embedding = update_embedding
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.clip_grad = clip_grad
        self.log_path = log_path
        self.optimizer = optimizer

    def build_graph(self):
        # 添加占位符
        self.add_placeholders()
        # 网络结构
        self.lookup_layer_op()
        self.biLSTM_layer_op()

        # 设置损失
        self.loss_op()
        self.trainstep_op()
        # self.init_op()

    def add_placeholders(self):
        with tf.variable_scope("placeholder"):
            self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
            self.dropout_pl = tf.placeholder(tf.float32, shape=[], name="dropout_pl")
            self.lr_pl = tf.placeholder(tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("embedding"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding, name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("Bi-LSTM"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                                cell_bw=cell_bw,
                                                                                inputs=self.word_embeddings,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

            W = tf.get_variable(name="Weight",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name="Bias",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        with tf.variable_scope("CRF_loss"):
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

    def trainstep_op(self):
        with tf.variable_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)


    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """
        将网络结构的计算图写入文件
        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.log_path, sess.graph)
