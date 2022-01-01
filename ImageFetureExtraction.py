import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
print(tf.__version__)

from keras.models import load_model
model = load_model('vehicle3.h5')


filepath=r"D:\Python\SmartVehicleGuidance\data\images3\test.png"
img_height = 60
img_width = 90

model.summary()


class_names = ['Car', 'Truck', 'otherStuff']

img=cv2.imread(filepath)

img= cv2.resize(img, (img_width,img_height),interpolation=cv2.INTER_AREA)
frame=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_array = tf.expand_dims(frame, 0) # Create a batch
start=time.time()
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
stop=time.time()

diff= stop-start
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

print("Time for processing "+str(diff))
