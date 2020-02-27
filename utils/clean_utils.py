import re


class Clean(object):
    def __init__(self):
        pass

    def cut_corpus(self):
        pass

    def deal_paddle(self):
        pass

    def rule_set(self):
        pass

    def item_eval(self):
        pass


class Clean_64_Corpus(Clean):
    pass


class Clean_Punc_Corpus(Clean):
    # 原始数据集路径
    data_path = ''
    # 清洗后带标签的原始数据集路径
    data_with_label_path = ''
    # 清洗后不带标签的原始数据集路径
    data_without_label_path = ''
    # 百度lac标注的数据集路径
    paddle_data_path = ''
    # 百度lac数据集处理后保存的路径
    paddle_data_path_save = ''
    # 规则匹配后原始数据集的路径
    data_path_save = ''
    # 标签
    tag_labels = ''
    # 转换成按字标注后数据集存储路径
    data_label_word = ''

    def _init_(self):
        pass

    def cut_corpus(self):
        char = ''
        tag = ''
        sents = []
        tags = []
        with open(self.data_path, encoding='utf-8')as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    char_, tag_ = line.strip().split()
                    # 去除括号
                    if char_ not in ['（', '）']:
                        char += char_
                        tag = tag + tag_ + ' '
                    # 判断是否为标点符号
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

        with open(self.data_with_label_path, 'w', encoding='utf-8') as fw:
            for sent, label in zip(sents, tags):
                # 判断是否为空
                if sent != '':
                    fw.write(sent + '|' + label + '\n')
        fw.close()

        with open(self.data_without_label_path, 'w', encoding='utf-8') as fwn:
            for sent in sents:
                # 判断是否为空
                if sent != '':
                    fwn.write(sent + '\n')

        fwn.close()

    def deal_paddle(self):
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        # 存储每行句子
        paddle_char = []
        # 存储每行标签
        paddle_label = []
        with open(self.paddle_data_path, encoding='utf-8')as f:
            lines = f.readlines()
            for line in lines:
                sent = ''
                label = ''
                items = re.findall(p1, line)
                """
                规则
                1.若百度lac有标注，原始数据集无标注，添加百度的标注
                2.哪个标注更长，就采用哪个方案
                3.百度ORG的优先级高于我们数据集的LOC PER 
                4.百度LOC的优先级高于我们数据集的ORG PER
                """
                for item in items:
                    if str(item)[0] != ',':
                        sent += str(item).split(',')[0]
                        if str(item).split(', ')[1] in ['nr', 'PER']:
                            label += 'B-PER' + ' '
                            for index in range(len(str(item).split(',')[0]) - 1):
                                label += 'I-PER' + ' '
                        elif str(item).split(', ')[1] in ['ns', 'LOC']:
                            label += 'B-LOC' + ' '
                            for index in range(len(str(item).split(',')[0]) - 1):
                                label += 'I-LOC' + ' '
                        elif str(item).split(', ')[1] in ['nt', 'ORG']:
                            label += 'B-ORG' + ' '
                            for index in range(len(str(item).split(',')[0]) - 1):
                                label += 'I-ORG' + ' '
                        else:
                            for index in range(len(str(item).split(',')[0])):
                                label += 'O' + ' '
                    else:
                        sent += str(item)[0]
                        label += 'O' + ' '
                paddle_char.append(sent)
                paddle_label.append(label)

        with open(self.paddle_data_path_save, 'w', encoding='utf-8') as fw:
            for sent_, label_ in zip(paddle_char, paddle_label):
                fw.write(sent_ + '|' + label_ + '\n')
        fw.close()

    def rule_set(self):
        label_t = []
        sent_t = []
        t = 0
        labels = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
        with open(self.paddle_data_path_save, encoding='utf-8') as f:
            with open(self.data_with_label_path, encoding='utf-8') as fr:
                lines_p = f.readlines()
                lines_t = fr.readlines()
                for line_p, line_t in zip(lines_p, lines_t):
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

        with open(self.data_path_save, 'w', encoding='utf-8') as fw:
            for sent_, label_ in zip(sent_t, label_t):
                fw.write(sent_ + '|' + label_ + '\n')
        fw.close()

    # 抽取实体，若无实体，设置为None
    def extract(self, data_path, tag_label):
        result_data = []
        num = 0
        flag = 0
        item = ''
        with open(data_path, encoding='utf-8')as f:
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

    def item_eval(self):
        total_items_paddle = 0
        total_items_test = 0
        for tag_label in self.tag_labels:
            paddle_data, num_paddle = self.extract(self.paddle_data_path_save, tag_label)
            test_data, num_test = self.extract(self, self.data_path_save, tag_label)
            correct_num = 0
            for i in range(len(test_data)):
                if test_data[i] != 'None':
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

    # 按句标注转换成按字标注
    def sent2word(self):
        words = []
        tags = []
        with open(self.data_path_save, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = []
                sent_, label_ = line.strip().split('|')
                data.append(label_.strip().split())
                for i in range(len(data[0])):
                    words.append(sent_[i])
                    tags.append(data[0][i])

        #按字写入
        with open(self.data_label_word, 'w', encoding='utf-8') as fw:
            for word, tag in zip(words, tags):
                fw.write(word + ' ' + tag + '\n')
                if word in ['。', '！']:
                    fw.write('\n')
        fw.close()
