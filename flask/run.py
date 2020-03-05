# from flask import send_file, send_from_directory
from flask import Flask, session, redirect, url_for, escape, request, render_template

app = Flask(__name__)


# post方法
@app.route("/login", methods=['post'])
def login():
    data = request.form.get('contents')
    return render_template('login.html', name=data)


if __name__ == '__main__':
    app.run()
