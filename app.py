from flask import Flask, jsonify, request
import numpy as np
from urllib.request import urlopen, Request
import matplotlib.pyplot as plt
from fer import FER
import tensorflow as tf
from flask_cors import CORS, cross_origin
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = tf.keras.models.load_model("model.h5")

@app.route('/get_drunk/', methods=['POST'])
def get_drunk():
    data = request.form.get("image")
    
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

    req = Request(data, headers=hdr)

    f = urlopen(req)
    img = plt.imread(f, 0)
    detector = FER(mtcnn=True)
    emotions = detector.detect_emotions(img)[0]['emotions']

    test = np.array([[ 0, emotions["angry"], 0, emotions["disgust"], emotions["fear"], emotions["happy"], emotions["neutral"], emotions["sad"], emotions["surprise"], 0 ]])
    prediction = model.predict(test)
    print(prediction)

    if prediction[0][0] >= 0.9:
        return jsonify({
            "drunk": True
        })
    else:
        return jsonify({
            "drunk": False
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1050)