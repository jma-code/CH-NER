import configparser
import logging

class Config(object):
    log_path = ''
    file_path = ''

    def __init__(self, class_name):
        self.class_name = class_name

    def load_config(self):
        pass

#数据处理
class ConfigProcess(Config):
    #训练集路径
    trainData_path = ''
    #测试集路径
    testData_path = ''
    #字典存储路径
    vocab_path = ''
    #维度
    embedding_dim = 1

    def _init_(self, class_name):
        # 类名
        self.class_name = class_name

    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.file_path)
        self.trainData_path = config.get(self.class_name, 'trainData_path')
        self.testData_path = config.get(self.class_name, 'testData_path')
        self.vocab_path = config.get(self.class_name, 'vocab_path')
        self.embedding_dim = config.get(self.class_name, 'embedding_dim')

#训练
class ConfigTrain(Config):
    # 模型存储路径
    store_path = ''
    # 训练集路径
    trainData_path = ''
    # 测试集路径
    testData_path = ''
    # 是否打乱数据集
    shuffle = True
    # 每次训练句子数
    batch_size = 0
    # 迭代次数
    epoch = 0
    # 学习率
    lr = 1
    # 梯度裁剪
    clip = 1
    # 优化器
    optimizer = ''
    # 保留概率
    dropout = 1
    # 维度
    embedding_dim = 1

    def _init_(self, class_name):
        # 类名
        self.class_name = class_name

    # 从配置文件中读取参数
    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.file_path)
        self.store_path = config.get(self.class_name, 'store_path')
        self.trainData_path = config.get(self.class_name, 'trainData_path')
        self.testData_path = config.get(self.class_name, 'testData_path')
        self.shuffle = config.get(self.class_name, 'shuffle')
        self.batch_size = config.get(self.class_name, 'batch_size')
        self.epoch = config.get(self.class_name, 'epoch')
        self.lr = config.get(self.class_name, 'lr')
        self.clip = config.get(self.class_name, 'clip')
        self.optimizer = config.get(self.class_name, 'optimizer')
        self.dropout = config.get(self.class_name, 'dropout')
        self.embedding_dim = config.get(self.class_name, 'embedding_dim')

#预测
class ConfigPredict(Config):
    # 模型路径
    model_path = ''
    # 字典路径
    vocab_path = ''
    # 模型名
    demo_model = ''
    # 维度
    embedding_dim = 1

    def _init_(self, class_name):
        # 类名
        self.class_name = class_name

    # 从配置文件中读取参数
    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.file_path)
        self.model_path = config.get(self.class_name, 'model_path')
        self.vocab_path = config.get(self.class_name, 'vocab_path')
        self.demo_model = config.get(self.class_name, 'demo_model')
        self.embedding_dim = config.get(self.class_name, 'embedding_dim')

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger