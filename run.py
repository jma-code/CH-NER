# -*- coding: utf-8 -*-
# @Time  : 2020/3/5 18:03
# @Author : sjw
# @Desc : ==============================================
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# ======================================================
# @Project : flask
# @FileName: run.py
# @Software: PyCharm
from flask import Flask, jsonify, render_template, request
import predict

app = Flask(__name__)


@app.route('/predict', methods=['post'])
def pred():
    PER, LOC, ORG = predict.run(request.json, True)
    str_per, str_loc, str_org = 'PER: ', 'LOC: ', 'ORG: '
    for item in PER:
        str_per += item + '、'
    for item in LOC:
        str_loc += item + '、'
    for item in ORG:
        str_org += item + '、'
    # print(str_per)
    return str_per + '\n' + str_loc + '\n' + str_org


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
