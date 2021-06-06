import os
from flask import Flask, request, render_template

import main

basedir = os.path.abspath(os.path.dirname(__file__))  # 定义一个根目录 用于保存图片用

app = Flask(__name__)


@app.route('/api', methods=['GET', 'POST'])
def editorData():
    # 获取图片文件
    img = request.files.get("file")

    # 定义一个图片存放的位置 存放在static下面
    path = basedir + "/static/img/"

    # 图片名称
    imgName = img.filename

    # 图片path和名称组成图片的保存路径
    file_path = path + imgName

    # 保存图片
    img.save(file_path)


    # url是图片的路径
    url = './static/img/' + imgName
    out = main.main(url)
    os.remove(url)
    return out


@app.route('/', methods=["POST", "GET"])
def united():
    return render_template("index.html")


if __name__ == '__main__':
    Net = main.Net
    app.run('0.0.0.0', )
