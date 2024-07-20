from flask import Flask, request, flash, redirect, render_template, url_for
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cv2
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load model and label encoder
model = load_model('model/saved_model.keras')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('model/classes.npy')

def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    label = None
    img_filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            try:
                upload_folder = 'static/uploads'
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)

                img = preprocess_image(file_path)
                if img is None:
                    flash('Error processing image')
                    return redirect(request.url)

                prediction = model.predict(img)
                label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                img_filename = file.filename

                return render_template('index.html', label=label, img_filename=img_filename)
            except Exception as e:
                flash(f'An error occurred: {e}')
                return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
