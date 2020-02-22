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
        num_tag = 0  # 统计tag_label总的实体数
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
                num_tag += len(per_item)
                per_items.append(per_item)
        return per_items, num_tag


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
        flag = 0
        num_tag = 0
        for line in lines:
            if line != '\n':
                char, tag = line.strip().split()
                if tag == 'B-' + tag_label:
                    if flag == 1:
                        sent_data.append(item)
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
                    num_tag += len(sent_data)
                else:
                    data.append('None')
                sent_data = []
        return data, num_tag


def comput_eval(paddle_path, test_path):
    tag_label = 'LOC'
    paddle_data, num_loc_paddle = paddle_item(paddle_path, tag_label)
    test_data, num_loc_test = test_item(test_path, tag_label)
    correct_loc_num = 0
    for i in range(len(test_data)):
        if test_data[i] != 'None':
            for j in range(len(test_data[i])):
                for k in range(len(paddle_data[i])):
                    if test_data[i][j] == paddle_data[i][k]:
                        correct_loc_num += 1
                        break
    print('LOC正确识别的实体数：', correct_loc_num)
    print('LOC总的识别实体数：', num_loc_paddle)
    print('LOC的正确率accurary:', float(correct_loc_num) / float(num_loc_paddle))
    print('LOC的召回率recall：', float(correct_loc_num) / float(num_loc_test))
    med_add = (float(correct_loc_num) / float(num_loc_paddle)) + (float(correct_loc_num) / float(num_loc_test))
    f_loc = 2 * float(correct_loc_num) / float(num_loc_test) * float(correct_loc_num) / float(num_loc_paddle)/med_add
    print('LOC的F测度值：', f_loc)

    tag_label = 'PER'
    paddle_data, num_per_paddle = paddle_item(paddle_path, tag_label)
    test_data, num_per_test = test_item(test_path, tag_label)
    correct_per_num = 0
    for i in range(len(test_data)):
        if test_data[i] != 'None':
            for j in range(len(test_data[i])):
                for k in range(len(paddle_data[i])):
                    if test_data[i][j] == paddle_data[i][k]:
                        correct_per_num += 1
                        break
    print('PER正确识别的实体数：', correct_per_num)
    print('PER总的识别实体数：', num_per_paddle)
    print('PER的正确率accurary:', float(correct_per_num) / float(num_per_paddle))
    med_add = (float(correct_per_num) / float(num_per_paddle)) + (float(correct_per_num) / float(num_per_test))
    f_per = 2 * float(correct_per_num) / float(num_per_test) * float(correct_per_num) / float(num_per_paddle) / med_add
    print('PER的F测度值：', f_per)

    tag_label = 'ORG'
    paddle_data, num_org_paddle = paddle_item(paddle_path, tag_label)
    test_data, num_org_test = test_item(test_path, tag_label)
    correct_org_num = 0
    for i in range(len(test_data)):
        if test_data[i] != 'None':
            for j in range(len(test_data[i])):
                for k in range(len(paddle_data[i])):
                    if test_data[i][j] == paddle_data[i][k]:
                        correct_org_num += 1
                        break
    print('ORG正确识别的实体数：', correct_org_num)
    print('ORG总的识别实体数：', num_org_paddle)
    print('ORG的正确率accurary:', float(correct_org_num) / float(num_org_paddle))
    med_add = (float(correct_org_num) / float(num_org_paddle)) + (float(correct_org_num) / float(num_org_test))
    f_org = 2 * float(correct_org_num) / float(num_org_test) * float(correct_org_num) / float(num_org_paddle) / med_add
    print('ORG的F测度值：', f_org)

    total_item_paddle = num_loc_paddle + num_org_paddle + num_per_paddle
    total_item_test = num_loc_test + num_org_test + num_per_test
    print('总的识别实体数：', total_item_paddle)
    print('总的实体数量：',total_item_test)


if __name__ == '__main__':

    comput_eval('lac_data.txt', 'test_data')



