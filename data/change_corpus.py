# -*- coding: utf-8 -*-
# @Time  : 2020/2/23 18:02
# @Author : sjw
# @Desc : ==============================================
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# ======================================================
# @Project : tensorflow_learnbyself
# @FileName: change_corpus.py
# @Software: PyCharm

def clean_kuohao(test_path):
    # 去掉括号
    with open(test_path, encoding='utf-8')as f:
        lines = f.readlines()
        sent = []
        label = []
        sents = []
        labels = []
        for line in lines:
            if line != '\n':
                char, tag = line.strip().split()
                if char not in ['(', ')', '（', '）']:
                    sent.append(char)
                    label.append(tag)
            else:
                sents.append(sent)
                labels.append(label)
                sent = []
                label = []
    with open(test_path, 'w', encoding='utf-8')as fw:
        for sent_, label_ in zip(sents, labels):
            for c_w, t_w in zip(sent_, label_):
                fw.write(c_w + '\t' + t_w + '\n')
            fw.write('\n')
        fw.close()


def change_64_corpus(test_path, test_64_path):
    """
    将长句子转换为短句子64，方便paddle预测
    :param test_path: 原始数据集的路径
    :param test_64_path: 修改后数据集的路径
    :return:
    """
    clean_kuohao(test_path)
    with open(test_path, encoding='utf-8')as f:
        with open(test_64_path, 'w', encoding='utf-8')as fw:
            lines = f.readlines()
            sentences = []  # 每次长度达到64或小于就存入
            labels = []  # 保存标签，方便后续输入文件，重新制作数据集
            length = 0  # 用来统计句子的长度信息用来切分
            sent_ = []  # 用来暂时存字的位置
            tag_ = []  # 用来暂时存标签的位置
            for line in lines:
                if line != '\n':
                    length += 1
                    char, tag = line.strip().split()
                    sent_.append(char)  # 下标从0开始
                    tag_.append(tag)  # 下标从0开始
                    if length == 64:
                        # 拆分句子，判断是否为B和O
                        if tag in ['B-PER', 'B-LOC', 'B-ORG', 'O']:
                            # 设置切点，将之前的句子输入到sentence中
                            # sentences.append(''.join(sent_[:length - 1]))
                            sentences.append(sent_[:length - 1])
                            labels.append(tag_[:length - 1])
                            # print(sentences)

                            # 考虑滑动窗口=5  将char改为sent_[length-5:length]
                            sent_, tag_ = sent_[length - 6:length], tag_[length - 6:length]

                            length = len(sent_)
                        else:
                            for i in range(length - 1, -1, -1):
                                if tag_[i] in ['B-PER', 'B-LOC', 'B-ORG', 'O']:
                                    sentences.append(sent_[:i])
                                    labels.append(tag_[:i])
                                    sent_, tag_ = sent_[i: length], tag_[i: length]  # 修改length + 1 为 length
                                    length = len(sent_)
                                    break
            if len(sent_) != 0:
                sentences.append(sent_)
                labels.append(tag_)
            # print(labels)
            # print(sentences)
            for sentence, label in zip(sentences, labels):
                for one_char, one_tag in zip(sentence, label):
                    fw.write(one_char + '\t' + one_tag + '\n')
                fw.write('\n')
            fw.close()


if __name__ == '__main__':
    change_64_corpus('wiki_test', 'wiki_64_test')
