import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
from flask import Flask, request, render_template, redirect
import cv2

model = tf.keras.models.load_model(
    filepath='rice.h5',
    custom_objects={'KerasLayer': hub.KerasLayer}
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'Data', 'val')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, f.filename)
        f.save(filepath)
        a2 = cv2.imread(filepath)
        a2 = cv2.resize(a2, (224, 224))
        a2 = np.array(a2) / 255.0
        a2 = np.expand_dims(a2, 0)
        pred = model.predict(a2)
        pred_class = pred.argmax()
        confidence = round(float(pred[0][pred_class]) * 100, 2)  # as percentage
        df_labels = {
            'arborio': 0,
            'basmati': 1,
            'ipsala': 2,
            'jasmine': 3,
            'karacadag': 4
        }
        prediction = None
        for i, j in df_labels.items():
            if pred_class == j:
                prediction = i
        # Optionally, remove the uploaded file after prediction
        # os.remove(filepath)
        return render_template('results.html', prediction_text=prediction, confidence=confidence)
    return redirect('/details')

if __name__ == "__main__":
    app.run(debug=True)