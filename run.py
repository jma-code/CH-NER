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
    return jsonify({'PER': PER, 'LOC': LOC, 'ORG': ORG})


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.config['JSON_AS_ASCII'] = False
    app.run()
