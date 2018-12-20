from flask import Flask, jsonify, render_template, request
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import random

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('scrathpad.html')

@app.route("/predict", methods=['POST'])
def predictLabel():
    img = request.form['img']
    img = img.split(';')[1].split(',')[1]
    img = base64.b64decode(img)
    img_bytes = io.BytesIO(img)
    im = Image.open(img_bytes)
    arr = np.array(im)
    plt.clf()
    plt.imshow(arr)
    plt.axis('off')
    fname = random.randint(0, 2000000000)
    plt.savefig('{}.png'.format(fname))
    os.system("python contourtest.py {}.png".format(fname))
    with open("{}.png_res".format(fname), "r") as f:
        res = f.readline()
    os.system("del {}.png {}.png_res".format(fname, fname))
    return jsonify({'status':'OK', 'res': res})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
