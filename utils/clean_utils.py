import re


class Clean(object):
    def __init__(self, corpus_path, paddle_output_path, corpus_after_clean_path):
        self.corpus_path = corpus_path
        self.paddle_output_path = paddle_output_path
        self.corpus_after_clean_path = corpus_after_clean_path

    def cut_corpus(self):
        pass

    def deal_paddle(self):
        pass

    def rule_set(self):
        pass

    def item_eval(self):
        pass


class Clean_64_Corpus(Clean):

    def __init__(self, corpus_path, corpus_64_path, paddle_input_tsv_path, paddle_output_path, corpus_after_clean_path):

        self.corpus_path = corpus_path   # 原始数据集路径
        self.corpus_64_path = corpus_64_path  # 处理成64字的短句子
        self.paddle_input_tsv_path = paddle_input_tsv_path  # 处理为paddle可以预测的格式tsv
        self.paddle_output_path = paddle_output_path  # paddle输出的lac_data.txt
        self.corpus_after_clean_path = corpus_after_clean_path  # 规则处理完保存数据集路径
        self.tag_label = None  # 目标标签
        self.other_label = None  # 其它同样类型的别名标签

    def clean_kuohao(self):
        # 去掉括号
        with open(self.corpus_path, encoding='utf-8')as f:
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
        with open(self.corpus_path, 'w', encoding='utf-8')as fw:
            for sent_, label_ in zip(sents, labels):
                for c_w, t_w in zip(sent_, label_):
                    fw.write(c_w + '\t' + t_w + '\n')
                fw.write('\n')
            fw.close()

    def cut_corpus(self):
        self.clean_kuohao()
        with open(self.corpus_path, encoding='utf-8')as f:
            with open(self.corpus_64_path, 'w', encoding='utf-8')as fw:
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

    def to_paddle_tsv(self):
        data = []
        with open(self.corpus_path, encoding='utf-8')as f:

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
        with open(self.paddle_input_tsv_path, 'w', encoding='utf-8')as fw:
            for sentence in data:
                fw.write(sentence + '\n')

        fw.close()

    def deal_paddle(self):
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        paddle_item = []
        # i = 0
        with open(self.paddle_output_path, encoding='utf-8')as f:
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
                    if len(item) == 0:
                        continue
                    if str(item).split(', ')[1] == self.tag_label or str(item).split(', ')[1] == self.other_label:
                        per_item.append(str(item).split(', ')[0])
                if len(per_item) == 0:
                    per_items.append('None')
                else:
                    num_tag += len(per_item)
                    per_items.append(per_item)
            return per_items, num_tag

    def deal_corpus(self):
        with open(self.corpus_path, encoding='utf-8')as f:
            lines = f.readlines()
            item = ''
            data = []
            sent_data = []
            flag = 0
            num_tag = 0
            for line in lines:
                if line != '\n':
                    char, tag = line.strip().split()
                    if tag == 'B-' + self.tag_label:
                        if flag == 1:
                            sent_data.append(item)
                        item = char
                        flag = 1
                    elif tag == 'I-' + self.tag_label:
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

    def rule_set(self):
        paddle_data, num_paddle = self.deal_paddle()
        test_data, num_test = self.deal_corpus()
        # print(paddle_data)
        # print(test_data)
        # 输出位置信息
        with open(self.paddle_input_tsv_path, encoding='utf-8')as f:
            with open(self.corpus_64_path, encoding='utf-8')as f_label:
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

                            str_label = 'I-' + self.tag_label
                            start_label = 'B-' + self.tag_label
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

        with open(self.corpus_after_clean_path, 'w', encoding='utf-8')as fw:
            # print(data_labels[32])
            # print(data_sentences[32])
            for i in range(len(data_sentences)):
                for c_w, t_w in zip(data_sentences[i], data_labels[i]):
                    fw.write(c_w + '\t' + t_w + '\n')
                fw.write('\n')

    def item_eval(self):
        tag_labels = ['LOC', 'PER', 'ORG']
        other_labels = ['ns', 'nr', 'nt']
        total_items_paddle = 0
        total_items_test = 0
        for tag_label, other_label in zip(tag_labels, other_labels):
            self.tag_label = tag_label
            self.other_label = other_label
            paddle_data, num_paddle = self.deal_paddle()
            test_data, num_test = self.deal_corpus()
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



class Clean_Punc_Corpus(Clean):

    def _init_(self, corpus_path, corpus_with_label_path, corpus_without_label_path, paddle_output_path, paddle_data_path_save, data_path_save, tag_labels, corpus_after_clean_path):
        # 原始数据集路径
        self.corpus_path = corpus_path
        # 清洗后带标签的原始数据集路径
        self.corpus_with_label_path = corpus_with_label_path
        # 清洗后不带标签的原始数据集路径
        self.corpus_without_label_path = corpus_without_label_path
        # 百度lac标注的数据集路径
        self.paddle_output_path = paddle_output_path
        # 百度lac数据集处理后保存的路径
        self.paddle_data_path_save = paddle_data_path_save
        # 规则匹配后原始数据集的路径
        self.data_path_save = data_path_save
        # 标签
        self.tag_labels = tag_labels
        # 转换成按字标注后数据集存储路径
        self.corpus_after_clean_path = corpus_after_clean_path


    def cut_corpus(self):
        char = ''
        tag = ''
        sents = []
        tags = []
        with open(self.corpus_path, encoding='utf-8')as f:
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

        with open(self.corpus_with_label_path, 'w', encoding='utf-8') as fw:
            for sent, label in zip(sents, tags):
                # 判断是否为空
                if sent != '':
                    fw.write(sent + '|' + label + '\n')
        fw.close()

        with open(self.corpus_without_label_path, 'w', encoding='utf-8') as fwn:
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
        with open(self.paddle_output_path, encoding='utf-8')as f:
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
            with open(self.corpus_with_label_path, encoding='utf-8') as fr:
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
        with open(self.corpus_after_clean_path, 'w', encoding='utf-8') as fw:
            for word, tag in zip(words, tags):
                fw.write(word + ' ' + tag + '\n')
                if word in ['。', '！']:
                    fw.write('\n')
        fw.close()
