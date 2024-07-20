import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        subpath = os.path.join(folder, subdir)
        for filename in os.listdir(subpath):
            img_path = os.path.join(subpath, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize to 128x128
                images.append(img)
                labels.append(subdir)
    return np.array(images), np.array(labels)

images, labels = load_images_from_folder("D:\DataMining_Sem6\dataset_kucing")

# Pembagian Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

#Encoding Label
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
y_train_enc = to_categorical(y_train_enc)
y_test_enc = to_categorical(y_test_enc)

# Simpan label encoder ke file 'classes.npy'
np.save('classes.npy', le.classes_)

#Bangun Model CNN dengan Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Latih Model
model.fit(X_train, y_train_enc, epochs=95, batch_size=55, validation_data=(X_test, y_test_enc))

#Evaluasi Model
loss, accuracy = model.evaluate(X_test, y_test_enc)
print(f'Accuracy: {accuracy * 100:.2f}%')

#save model
model.save('saved_model.keras')
model = load_model('saved_model.keras')
