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

    # 按句子写入文档 （让句子按照逗号拆分）
    with open(write_path, 'w', encoding='utf-8')as fw:
        for sentence in data:
            fw.write(sentence + '\n')

    fw.close()


# read_corpus_topaddle('test_data', 'data_paddle.tsv')

import re


def paddle_item(paddle_path, tag_label, other_label):
    '''
    将Paddle处理出来的数据进行统计，没有对应实体的补None
    :param other_label: 百度中使用的置信度低的标签[nr/ns/nt PER/LOC/ORG]
    :param paddle_path: Paddle输出的文件路径
    :param tag_label: 目标标签
    :return:
    '''
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    paddle_item = []
    # i = 0
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
            # 输出序号
            # i += 1
            # print(items, i)
            for item in items:
                if len(item) == 0: continue
                if str(item).split(', ')[1] == tag_label or str(item).split(', ')[1] == other_label:
                    per_item.append(str(item).split(', ')[0])
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
                    if flag == 1:
                        sent_data.append(item)  # 添加情况：BIBI
                        item = ''
                        flag = 0

                    data.append(sent_data)
                    num_tag += len(sent_data)
                else:
                    data.append('None')
                sent_data = []
        return data, num_tag


def comput_eval(paddle_path, test_path):
    """
    计算各种测评值
    :param paddle_path: paddle输出实体的路径
    :param test_path: 原始测试数据集的路径
    :return:
    """
    tag_labels = ['LOC', 'PER', 'ORG']
    other_labels = ['ns', 'nr', 'nt']
    total_items_paddle = 0
    total_items_test = 0
    for tag_label, other_label in zip(tag_labels, other_labels):
        paddle_data, num_paddle = paddle_item(paddle_path, tag_label, other_label)
        test_data, num_test = test_item(test_path, tag_label)
        correct_num = 0
        for i in range(len(test_data)):
            if test_data[i] != 'None':
                # if tag_label == 'ORG':
                #     print(test_data[i], paddle_data[i], i + 1)
                for j in range(len(test_data[i])):
                    for k in range(len(paddle_data[i])):
                        if test_data[i][j] == paddle_data[i][k]:
                            correct_num += 1
                            break
        print(tag_label + '正确识别的实体数：', correct_num)
        print(tag_label + '总的识别实体数：', num_paddle)
        print(tag_label + '的正确率accuracy:', float(correct_num) / float(num_paddle))
        print(tag_label + '的召回率recall：', float(correct_num) / float(num_test))
        med_add = (float(correct_num) / float(num_paddle)) + (float(correct_num) / float(num_test))
        f_loc = 2 * float(correct_num) / float(num_test) * float(correct_num) / float(num_paddle) / med_add
        print(tag_label + '的F测度值：', f_loc)

        total_items_paddle += num_paddle
        total_items_test += num_test
    print('总的识别实体数：', total_items_paddle)
    print('总的实体数量：', total_items_test)


def change_label(test_path,paddle_input_path,  paddle_output_path, test_clean_path, tag_label, other_label):
    paddle_data, num_paddle = paddle_item(paddle_output_path, tag_label, other_label)
    test_data, num_test = test_item(test_path, tag_label)
    # print(paddle_data)
    # print(test_data)
    # 输出位置信息
    with open(paddle_input_path, encoding='utf-8')as f:
        with open(test_path, encoding='utf-8')as f_label:
            lines = f.readlines()
            data = f_label.readlines()
            data_sentences = []
            data_labels = []
            data_sentence = []
            data_label = []
            for data_ in data:
                if data_ != '\n':
                    sent_, tag_ = data_.strip().split()
                    data_sentence.append(sent_)
                    data_label.append(tag_)
                else:
                    data_sentences.append(data_sentence)
                    data_labels.append(data_label)
                    data_sentence = []
                    data_label = []
            # print(data_sentences[0][1])  # 先将文件中的内容读出来，然后再写进去
            # print(data_labels[0])
            for i in range(len(paddle_data)):
                # 将单字实体删除
                if test_data[i] != 'None':
                    for item_i in test_data[i]:
                        if len(item_i) == 1:
                            index_test = lines[i].find(item_i)  # 单字实体对应的位置信息
                            data_labels[i][index_test] = 'O'


                if paddle_data[i] != 'None':
                    # if i == 538:
                    # print(i)
                    # print(paddle_data[i])  # 输出预测实体列表
                    # print(test_data[i])  # 输出实体列表
                    #
                    # print(len(paddle_data[i]))  # 输出实体长度
                    # print(lines[i])  # 输出对应句子



                    index_bias = 0
                    for item in paddle_data[i]:
                        item_len = len(item)
                        # print(lines[i][index_bias:])

                        index_paddle = lines[i][index_bias:].find(item)  # 实体对应的位置信息
                        # index_test = lines[i][index_test + item_len:].find(test_data[i][0])
                        # print(index_paddle + index_bias)  # 预测位置
                        # print(index_test)  # 原始位置

                        # print(data_sentences[i][index_paddle + index_bias])  # 输出在原始数据集里的位置的字符（预测位置）
                        # print(data_labels[i][index_paddle + index_bias])  # 输出位置处的标签（预测位置）

                        loc = data_labels[i][index_paddle + index_bias]

                        # print(data_sentences[i][index_test])  # 输出原始位置处的字符（原始位置）
                        # print(data_labels[i][index_test])  # 输出原始位置处的标签（原始位置）

                        str_label = 'I-' + tag_label
                        start_label = 'B-' + tag_label
                        # 遇到B，判断之后I的长度。谁长选谁。（记录之后由B改为I的位置，进行分析）
                        if loc == start_label:
                            for change_i in range(1, item_len):

                                data_labels[i][index_paddle + index_bias + change_i] = str_label

                        # 遇到I，判断之后I的长度，谁长选谁。
                        if loc == str_label:
                            for change_i in range(1, item_len):
                                data_labels[i][index_paddle + index_bias + change_i] = str_label

                        # 遇到O，改为B，后边添加I的长度。
                        if loc == 'O':
                            data_labels[i][index_paddle + index_bias] = start_label
                            for change_i in range(1, item_len):
                                data_labels[i][index_paddle + index_bias + change_i] = str_label

                        # B-ORG遇到B-LOC
                        else:
                            data_labels[i][index_paddle + index_bias] = start_label
                            for change_i in range(1, item_len):
                                data_labels[i][index_paddle + index_bias + change_i] = str_label

                        index_bias = index_paddle + item_len
                        # print(data_labels[i])
                    # break

    with open(test_clean_path, 'w', encoding='utf-8')as fw:
        # print(data_labels[32])
        # print(data_sentences[32])
        for i in range(len(data_sentences)):
            for c_w, t_w in zip(data_sentences[i], data_labels[i]):
                fw.write(c_w + '\t' + t_w + '\n')
            fw.write('\n')


if __name__ == '__main__':
    # read_corpus_topaddle('test_64_data', 'data_64_paddle.tsv')
    # change_label('test_64_data','data_64_paddle.tsv', 'lac_data.txt', 'test_64_clean_data', 'PER', 'nr')
    # change_label('test_64_clean_data', 'data_64_paddle.tsv', 'lac_data.txt', 'test_64_clean_data', 'LOC', 'ns')
    # change_label('test_64_clean_data', 'data_64_paddle.tsv', 'lac_data.txt', 'test_64_clean_data', 'ORG', 'nt')
    # comput_eval('lac_data.txt', 'test_64_clean_data')

    #read_corpus_topaddle('wiki_64_test', 'data_64_paddle.tsv')
    change_label('wiki_64_test', 'data_64_paddle.tsv', 'lac_data.txt', 'wiki_64_clean_data', 'PER', 'nr')
    change_label('wiki_64_clean_data', 'data_64_paddle.tsv', 'lac_data.txt', 'wiki_64_clean_data', 'LOC', 'ns')
    change_label('wiki_64_clean_data', 'data_64_paddle.tsv', 'lac_data.txt', 'wiki_64_clean_data', 'ORG', 'nt')
    comput_eval('lac_data.txt', 'wiki_64_clean_data')
