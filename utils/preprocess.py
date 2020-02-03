
tag2label = {
    "O": 0,
    "B-PER": 1,"I-PER": 2,
    "B-LOC": 3,"I-LOC": 4,
    "B-ORG": 5,"I-ORG": 6,
}

def read_corpus(corpus_path):
    """
    读取数据集返回list
    :param corpus_path:
    :return:
    """
    data=[]
    with open(corpus_path, encoding='utf-8')as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:

