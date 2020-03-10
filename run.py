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
    PER_mess, LOC_mess, ORG_mess = predict.run(request.json, True)
    PER_result, LOC_result, ORG_result = [], [], []
    for p in PER_mess.values():
        PER_result.append(p)
    for l in LOC_mess.values():
        LOC_result.append(l)
    for o in ORG_mess.values():
        ORG_result.append(o)

    return jsonify({'PER': PER_result, 'LOC': LOC_result, 'ORG': ORG_result})


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.config['JSON_AS_ASCII'] = False
    app.run()
