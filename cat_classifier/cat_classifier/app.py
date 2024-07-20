from flask import Flask, request, render_template
from keras.models import load_model
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and label encoder
model = load_model('saved_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = 'static/' + file.filename
        file.save(file_path)
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        label = label_encoder.inverse_transform([np.argmax(prediction)])
        return render_template('index.html', label=label[0], img_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)  # Ubah menjadi app.run(debug=False) atau app.run() untuk produksi
