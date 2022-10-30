import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import json
import pathlib
print(tf.__version__)


from keras.models import load_model
model = load_model(r'models/vehicles_full_narrow.h5')
model.summary()
class_names = ['ambulance', 'car', 'truck']
img_height = 120
img_width = 60


def class_extraction(d_object):
    d_object=cv2.resize(d_object, (img_width,img_height),interpolation=cv2.INTER_AREA)
    d_object=cv2.cvtColor(d_object, cv2.COLOR_BGR2RGB)
    img_array = tf.expand_dims(d_object, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    objectDetected= class_names[np.argmax(score)]
    confi =  100 * np.max(score)
    
    return objectDetected, confi

if __name__=="__main__":
    testPath=r"D:\Python\SmartVehicleGuidance\data\truck\test.jpg"
    image=cv2.imread(testPath)
    print(class_extraction(image))
    
