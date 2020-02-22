# -*- coding: utf-8 -*-
# @Time  : 2020/2/21 17:18
# @Author : sjw
# @Desc : ==============================================
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# ======================================================
# @Project : tensorflow_learnbyself
# @FileName: to_paddle_data.py
# @Software: PyCharm

def read_corpus_topaddle(test_path, write_path):
    '''
    读取数据集中的测试集，改变为paddle lac可以测试得数据集格式
    :param test_path: 测试集路径
    :param write_path: 写入文件的路径
    :return:
    '''
    # 读取数据，分成独立的句子
    data = []
    with open(test_path, encoding='utf-8')as f:

        lines = f.readlines()
        sent = ''
        for line in lines:
            if line != '\n':
                sent_, _ = line.strip().split()
                sent += sent_
            else:
                data.append(sent)
                sent = ''
    # print(data)

    # 按句子写入文档
    with open(write_path, 'w', encoding='utf-8')as fw:
        for sentence in data:
            fw.write(sentence + '\n')

    fw.close()


# read_corpus_topaddle('test_data', 'data_paddle.tsv')

import re


def paddle_item(paddle_path, tag_label):
    '''
    将Paddle处理出来的数据进行统计，没有对应实体的补None
    :param paddle_path: Paddle输出的文件路径
    :param tag_label: 目标标签
    :return:
    '''
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    paddle_item = []
    with open(paddle_path, encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            paddle_item.append(re.findall(p1, line))

        per_items = []
        for items in paddle_item:
            if len(items[0]) == 0:
                # 单独识别括号），则实体里边为空
                per_items.append('None')
                continue
            per_item = []
            for item in items:
                if len(item) == 0: continue
                if str(item).split(', ')[1] == tag_label:
                    per_item.append(str(item).split(',')[0])
            if len(per_item) == 0:
                per_items.append('None')
            else:
                per_items.append(per_item)
        return per_items


def test_item(test_path, tag_label):
    '''
    将测试集中的对应实体抽出，没有实体的为None
    :param test_path: 测试集路径
    :param tag_label: 需要抽取的标签值
    :return:
    '''
    with open(test_path, encoding='utf-8')as f:
        lines = f.readlines()
        item = ''
        data = []
        sent_data = []
        for line in lines:
            if line != '\n':
                flag = 0
                char, tag = line.strip().split()
                if tag == 'B-' + tag_label:
                    if flag == 1:
                        sent_data.append(item)
                        item = ''
                    item = char
                    flag = 1
                elif tag == 'I-' + tag_label:
                    item += char
                elif len(item) != 0:
                    sent_data.append(item)
                    item = ''
                    flag = 0
            else:
                if len(sent_data) != 0:
                    data.append(sent_data)
                else:
                    data.append('None')
                sent_data = []
        return data



if __name__ == '__main__':
    tag_label = 'LOC'
    paddle_data = paddle_item('lac_data.txt', tag_label)
    test_data = test_item('test_data', tag_label)
    print(paddle_data)
    print(test_data)
    num = 0
    for items in paddle_data:
        if items != 'None':
            num += 1
    print('paddle识别出的实体数量：', num)
    num = 0
    for items in test_data:
        if items != 'None':
            num += 1
    print('总的实体数量：', num)
    print(len(paddle_data), len(test_data))
