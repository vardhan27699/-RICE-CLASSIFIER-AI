# Importing necessary libraries
# Building deep learning models
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import pathlib

# For separating train and test sets
from sklearn.model_selection import train_test_split
# For visualizations
import matplotlib.pyplot as plt
import matplotlib.image as img
import PIL.Image as Image

data_dir = "Rice_Image_Dataset"
data_dir = pathlib.Path(data_dir)


arborio = list(data_dir.glob('Arborio/*')) [:600]
basmati = list(data_dir.glob('Basmati/*')) [:600]
ipsala = list(data_dir.glob('Ipsala/*')) [:600]
jasmine = list(data_dir.glob('Jasmine/*')) [:600]
karacadag = list(data_dir.glob('Karacadag/*')) [:600]

fig, ax = plt.subplots(ncols=5, figsize=(20,5))
fig.suptitle('Rice Category')
arborio_image = img.imread(arborio [0])
basmati_image = img.imread(basmati [0])
ipsala_image = img.imread(ipsala[0])
jasmine_image = img.imread(jasmine[0])
karacadag_image = img.imread(karacadag[0])
ax[0].set_title('arborio')
ax[1].set_title('basmati')
ax[2].set_title('ipsala')
ax[3].set_title('jasmine')
ax[4].set_title('karacadag')
ax[0].imshow(arborio_image)
ax[1].imshow(basmati_image)
ax[2].imshow(ipsala_image)
ax[3].imshow(jasmine_image)
ax[4].imshow(karacadag_image)

# Contains the images path
df_images = {
'arborio': arborio,
'basmati': basmati,
'ipsala': ipsala,
'jasmine': jasmine,
'karacadag': karacadag
}

# Contains numerical labels for the categories
df_labels = {
'arborio' : 0,
'basmati': 1,
'ipsala': 2,
'jasmine': 3,
'karacadag': 4
}

img = cv2.imread(str(df_images['arborio'][0]))
resized_img = cv2.resize(img, (224, 224))
# print(img.shape)  # Optional: for debugging

X, y = [], [] # X = images, y = labels
for label, images in df_images.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (224, 224))
        X.append(resized_img)
        y.append(df_labels[label])

# Standarizing
X = np.array(X)
X = X / 255
y = np.array(y)

# Separating data into training, test and validation sets
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

mobile_net_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
mobile_net_layer = hub.KerasLayer(
    mobile_net_url, input_shape=(224, 224, 3), trainable=False)

num_label = 5
model = tf.keras.Sequential([
    mobile_net_layer,
    tf.keras.layers.Dense(num_label, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)
model.save('rice.h5')

from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import pandas as pd
init_notebook_mode(connected=True)
acc = pd.DataFrame({'train': history.history['accuracy'], 'val': history.history['val_accuracy']})
fig = px.line(acc, x=acc.index, y=acc.columns [0::], title='Training and Evaluation Accuracy every Epoch', markers=True)
fig.show()

loss = pd.DataFrame({'train': history.history['loss'], 'val': history.history['val_loss']})
fig = px.line (loss, x=loss.index, y=loss.columns[0::], title='Training and Evaluation Loss every Epoch', markers=True)
fig.show()

a1 = cv2.imread("../input/rice-image-dataset/Rice_Image_Dataset/Basmati/basmati (10).jpg")
a1 = cv2.resize(a1, (224,224))
a1 = np.array(a1)
a1 =a1/255
a1 = np.expand_dims(a1, 0)
pred = model.predict(a1)
pred[0]
pred[0].argmax()
pred[0]

predicted_class = pred[0].argmax()
for i, j in df_labels.items():
    if predicted_class == j:
        print(i)
model.save('rice.h5')