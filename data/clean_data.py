import re

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

def comput_eval(paddle_path, test_path):
    """
    :param paddle_path:百度标注文件路径
    :param test_path: 括测试标注文件路径
    :return:
    """
    tag_label = 'LOC'
    paddle_data, num_loc_paddle = extract(paddle_path, tag_label)
    test_data, num_loc_test = extract(test_path, tag_label)
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
    paddle_data, num_per_paddle = extract(paddle_path, tag_label)
    test_data, num_per_test = extract(test_path, tag_label)
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
    print('PER的召回率recall：', float(correct_per_num) / float(num_per_test))
    med_add = (float(correct_per_num) / float(num_per_paddle)) + (float(correct_per_num) / float(num_per_test))
    f_per = 2 * float(correct_per_num) / float(num_per_test) * float(correct_per_num) / float(num_per_paddle) / med_add
    print('PER的F测度值：', f_per)

    tag_label = 'ORG'
    paddle_data, num_org_paddle = extract(paddle_path, tag_label)
    test_data, num_org_test = extract(test_path, tag_label)
    correct_org_num = 0
    for i in range(len(test_data)):
        if test_data[i] != 'None':
            print(test_data[i], paddle_data[i], i + 1)
            for j in range(len(test_data[i])):
                for k in range(len(paddle_data[i])):
                    if test_data[i][j] == paddle_data[i][k]:
                        correct_org_num += 1
                        break
    print('ORG正确识别的实体数：', correct_org_num)
    print('ORG总的识别实体数：', num_org_paddle)
    print('ORG的正确率accurary:', float(correct_org_num) / float(num_org_paddle))
    print('ORG的召回率recall：', float(correct_org_num) / float(num_org_test))
    med_add = (float(correct_org_num) / float(num_org_paddle)) + (float(correct_org_num) / float(num_org_test))
    f_org = 2 * float(correct_org_num) / float(num_org_test) * float(correct_org_num) / float(num_org_paddle) / med_add
    print('ORG的F测度值：', f_org)

    total_item_paddle = num_loc_paddle + num_org_paddle + num_per_paddle
    total_item_test = num_loc_test + num_org_test + num_per_test
    print('总的识别实体数：', total_item_paddle)
    print('总的实体数量：',total_item_test)


def match_label(paddle_path, test_path, test_save_path):
    label_t = []
    sent_t = []
    with open(paddle_path, encoding = 'utf-8') as f:
        with open(test_path, encoding = 'utf-8') as fr:
            lines_p = f.readlines()
            lines_t = fr.readlines()
            for line_p,line_t in zip(lines_p,lines_t):
                tag = ''
                data_p = []
                data_t = []
                sent_, label_ = line_p.strip().split('|')
                sent, label = line_t.strip().split('|')
                data_p.append(label_.strip().split())
                data_t.append(label.strip().split())
                for i in range(len(data_p[0])):
                    if data_p[0][i] == 'B-PER' and data_t[0][i] == 'O':
                        tag = tag + 'B-PER' + ' '
                    elif data_p[0][i] == 'I-PER' and data_t[0][i] == 'O':
                        tag = tag + 'I-PER' + ' '
                    elif data_p[0][i] == 'I-PER' and data_t[0][i] == 'B-PER':
                        tag = tag + 'I-PER' + ' '
                    elif data_p[0][i] == 'B-ORG' and data_t[0][i] == 'O':
                        tag = tag + 'B-ORG' + ' '
                    elif data_p[0][i] == 'I-ORG' and data_t[0][i] == 'O':
                        tag = tag + 'I-ORG' + ' '
                    elif data_p[0][i] == 'I-ORG' and data_t[0][i] == 'B-ORG':
                        tag = tag + 'I-ORG' + ' '
                    elif data_p[0][i] == 'B-LOC' and data_t[0][i] == 'O':
                        tag = tag + 'B-LOC' + ' '
                    elif data_p[0][i] == 'I-LOC' and data_t[0][i] == 'O':
                        tag = tag + 'I-LOC' + ' '
                    elif data_p[0][i] == 'I-LOC' and data_t[0][i] == 'B-LOC':
                        tag = tag + 'I-LOC' + ' '
                    elif data_p[0][i] == 'B-ORG' and data_t[0][i] == 'B-LOC':
                        tag = tag + 'B-ORG' + ' '
                    elif data_p[0][i] == 'I-ORG' and data_t[0][i] == 'I-LOC':
                        tag = tag + 'I-ORG' + ' '
                    else:
                        tag = tag + data_t[0][i] + ' '
                label_t.append(tag)
                sent_t.append(sent)

    with open(test_save_path, 'w', encoding='utf-8') as fw:
        for sent_, label_ in zip(sent_t, label_t):
            fw.write(sent_ + '|'+label_ + '\n')
    fw.close()



if __name__ == '__main__':
    #clean_data('test_data', 'msra_label.tsv', 'infer.tsv')
    #clean_lac('lac_data_msra.txt', 'lac_data_msra_clean.tsv')
    #match_label('lac_data_msra_clean.tsv', 'msra_label.tsv', 'msra_label_sec.tsv')
    comput_eval('lac_data_msra_clean.tsv', 'msra_label_sec.tsv')