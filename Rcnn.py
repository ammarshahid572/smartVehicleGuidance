#using CNN with edge detection + contours


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
model = load_model('vehicle.h5')

minArea=20000
filepath=r"D:\Python\SmartVehicleGuidance\data\images3\test.png"
img_height = 120
img_width = 180
model.summary()


class_names = ['Car', 'Truck', 'otherStuff']


img=cv2.imread(filepath)

#Uncomment to use Live Video Feed
#cap= cv2.VideoCapture(1)
Once=True
while Once:
    start=time.time()
    #Uncomment to use a single file
    img=cv2.imread(filepath)

    #Uncomment to use live video Feed
    #_,img=cap.read()
    img= cv2.resize(img, (500,500),interpolation=cv2.INTER_AREA)
    canny= cv2.Canny(img, 50, 100)
    kernel = np.ones((3,3), np.uint8)
    #eroded = cv2.erode(canny, kernel, iterations=1)
    dilated = cv2.dilate(canny, kernel, iterations=2)
    cv2.imshow("binary", dilated)
    
    contours, hierarchies= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i,contour in enumerate(contours):
        if (hierarchies[0][i][3])<0:
            (x,y,w,h) = cv2.boundingRect(contour)
            if w*h > minArea:
                    d_object=img[y:y+h,x:x+w]
                    d_object=cv2.resize(d_object, (img_width,img_height),interpolation=cv2.INTER_AREA)
                    d_object=cv2.cvtColor(d_object, cv2.COLOR_BGR2RGB)
                    img_array = tf.expand_dims(d_object, 0)
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
                    img = cv2.putText(img, "{} {:.2f}".format(class_names[np.argmax(score)], 100 * np.max(score)), (x,y+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
    end=time.time()
    fps= str(int(1/(end-start)))
    img = cv2.putText(img, fps, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4, cv2.LINE_AA)
    cv2.imshow("Window",img)
    k=cv2.waitKey(1)
    if k & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
             
