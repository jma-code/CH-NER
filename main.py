# -*- coding: utf-8 -*-
# @Time  : 2020/2/16 15:19
# @Author :
# @Desc : ==============================================
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# ======================================================
# @Project : LSTM_CRF_IE
# @FileName: main.py.py
# @Software: PyCharm
import argparse
import train
import predict


parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--mode', type=str, default='predict', help='train/test/predict')
args = parser.parse_args()

if args.mode == 'predict':
    #predict.run()
    PER_mess, LOC_mess, ORG_mess = predict.run('我在北京上北京大学,周恩来是中国总理,我喜欢北京。我在清华大学，毛泽东是中国主席，他去过苏联。', True)
    print(PER_mess, LOC_mess, ORG_mess)
else:
    train.run(args.mode)

