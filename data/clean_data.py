import re

#清洗数据
def clean_data(data_path, data_with_label, data_without_label):
    """
    :param data_path:原文件路径
    :param data_with_label: 清洗后带标签数据的路径
    :param data_without_label: 清洗后不带标签数据的路径
    :return:
    """
    char = ''
    tag = ''
    sents =[]
    tags = []
    with open(data_path, encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                char_, tag_ = line.strip().split()
                #去除括号
                if char_ not in ['（', '）']:
                    char += char_
                    tag = tag + tag_ + ' '
                #判断是否为标点符号
                if char_ in ['，', '。', '!']:
                    sents.append(char)
                    tags.append(tag)
                    char = ''
                    tag = ''
            else:
                sents.append(char)
                tags.append(tag)
                char = ''
                tag = ''

    with open(data_with_label, 'w', encoding='utf-8') as fw:
        for sent, label in zip(sents, tags):
            #判断是否为空
            if sent != '':
                fw.write(sent + '|'+label + '\n')
    fw.close()

    with open(data_without_label, 'w', encoding='utf-8') as fwn:
        for sent in sents:
            #判断是否为空
            if sent != '':
                fwn.write(sent + '\n')

    fwn.close()

#清洗百度LAC标注
def clean_lac(lac_data, lac_clean_data):
    """
    :param lac_data: 百度数据集
    :param lac_clean_data: 清洗后的数据集
    :return:
    """
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    #存储每行句子
    paddle_char = []
    #存储每行标签
    paddle_label = []
    with open(lac_data, encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            sent = ''
            label = ''
            items = re.findall(p1, line)
            for item in items:
                if str(item)[0] != ',':
                    sent += str(item).split(',')[0]
                    if str(item).split(', ')[1] in ['nr', 'PER']:
                        label += 'B-PER'+' '
                        for index in range(len(str(item).split(',')[0])-1):
                            label += 'I-PER'+' '
                    elif str(item).split(', ')[1] in ['ns', 'LOC']:
                        label += 'B-LOC'+' '
                        for index in range(len(str(item).split(',')[0])-1):
                            label += 'I-LOC'+' '
                    elif str(item).split(', ')[1] in ['nt', 'ORG']:
                        label += 'B-ORG'+' '
                        for index in range(len(str(item).split(',')[0])-1):
                            label += 'I-ORG'+' '
                    else:
                        for index in range(len(str(item).split(',')[0])):
                            label += 'O' + ' '
                else:
                    sent += str(item)[0]
                    label += 'O' + ' '
            paddle_char.append(sent)
            paddle_label.append(label)

    with open(lac_clean_data, 'w', encoding='utf-8') as fw:
        for sent_, label_ in zip(paddle_char, paddle_label):
            fw.write(sent_ + '|' + label_ + '\n')
    fw.close()

#抽取实体
def extract(path, tag_label):
    """
    :param path:百度标注数据集路径
    :param tag_label: 标签
    :return:
    """
    result_data = []
    num = 0
    flag = 0
    item = ''
    with open(path, encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            sent_data = []
            data = []
            sent_, label_ = line.strip().split('|')
            data.append(label_.strip().split())
            for i in range(len(data[0])):
                if data[0][i] == 'B-' + tag_label:
                    num += 1
                    if flag == 1:
                        sent_data.append(item)
                    item = sent_[i]
                    flag = 1
                elif data[0][i] == 'I-' + tag_label:
                    item += sent_[i]
                elif len(item) != 0:
                    sent_data.append(item)
                    item = ''
                    flag = 0
            if len(sent_data) != 0:
                result_data.append(sent_data)
            else:
                result_data.append('None')
    return result_data, num

#抽取实体，评估
def comput_eval(paddle_path, test_path):
    """
    计算各种测评值
    :param paddle_path: paddle输出实体的路径
    :param test_path: 原始测试数据集的路径
    :return:
    """
    tag_labels = ['LOC', 'PER', 'ORG']
    total_items_paddle = 0
    total_items_test = 0
    for tag_label in tag_labels:
        paddle_data, num_paddle = extract(paddle_path, tag_label)
        test_data, num_test = extract(test_path, tag_label)
        correct_num = 0
        for i in range(len(test_data)):
            if test_data[i] != 'None':
                if tag_label == 'LOC':
                    print(test_data[i], paddle_data[i], i + 1)
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

#修改标注
def match_label(paddle_path, test_path, test_save_path):
    """
    :param paddle_path:百度数据集
    :param test_path: 测试数据集
    :param test_save_path: 保存结果
    :return:
    """
    label_t = []
    sent_t = []
    t = 0
    labels = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
    with open(paddle_path, encoding = 'utf-8') as f:
        with open(test_path, encoding = 'utf-8') as fr:
            lines_p = f.readlines()
            lines_t = fr.readlines()
            for line_p,line_t in zip(lines_p,lines_t):
                t = t + 1
                tag = ''
                data_p = []
                data_t = []
                sent_, label_ = line_p.strip().split('|')
                sent, label = line_t.strip().split('|')
                data_p.append(label_.strip().split())
                data_t.append(label.strip().split())
                for i in range(len(data_p[0])):
                    if data_p[0][i] in labels[:6] and data_t[0][i] == 'O':
                        tag = tag + data_p[0][i] + ' '
                    elif data_p[0][i] == labels[1] and data_t[0][i] == labels[0]:
                        tag = tag + labels[1] + ' '
                    elif data_p[0][i] == labels[3] and data_t[0][i] == labels[2]:
                        tag = tag + labels[3] + ' '
                    elif data_p[0][i] == labels[5] and data_t[0][i] == labels[4]:
                        tag = tag + labels[5] + ' '
                    elif data_p[0][i] == labels[2] and data_t[0][i] in [labels[0], labels[4]]:
                        tag = tag + labels[2] + ' '
                    elif data_p[0][i] == labels[3] and data_t[0][i] in [labels[1], labels[5]]:
                        tag = tag + labels[3] + ' '
                    elif data_p[0][i] == labels[4] and data_t[0][i] in [labels[0], labels[2]]:
                        tag = tag + labels[4] + ' '
                    elif data_p[0][i] == labels[5] and data_t[0][i] in [labels[1], labels[3]]:
                        tag = tag + labels[5] + ' '
                    else:
                        tag = tag + data_t[0][i] + ' '
                label_t.append(tag)
                sent_t.append(sent)

    with open(test_save_path, 'w', encoding='utf-8') as fw:
        for sent_, label_ in zip(sent_t, label_t):
            fw.write(sent_ + '|'+label_ + '\n')
    fw.close()

#修改标注形式，改成单字标注
def sent2word(test_path, test_save_path):
    """
    :param test_path:原文件路径
    :param test_save_path: 存储路径
    :return:
    """
    words = []
    tags = []
    with open(test_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = []
            sent_, label_ = line.strip().split('|')
            data.append(label_.strip().split())
            for i in range(len(data[0])):
                words.append(sent_[i])
                tags.append(data[0][i])

    with open(test_save_path, 'w', encoding='utf-8') as fw:
        for word, tag in zip(words, tags):
            fw.write(word + ' ' + tag + '\n')
            if word in ['。', '！']:
                fw.write('\n')
    fw.close()


if __name__ == '__main__':
    #clean_data('test_data', 'msra_label.tsv', 'infer.tsv')
    #clean_lac('lac_data_msra.txt', 'lac_data_msra_clean.tsv')
    #match_label('lac_data_clean.tsv', 'test_label.tsv', 'test_label_thr.tsv')
    #comput_eval('lac_data_clean.tsv', 'test_label_thr.tsv')
    sent2word('test_label_thr.tsv', 'wiki_data')