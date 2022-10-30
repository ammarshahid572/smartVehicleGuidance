import cv2
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.models import load_model
model = load_model('vehicle.h5')
class_names = ['Car', 'Truck', 'otherStuff']

file=r"D:\Python\SmartVehicleGuidance\test2.jpg"
minArea=30000
img_height = 120
img_width = 180
model.summary()
image = cv2.imread(file)
# initialize OpenCV's selective search implementation and set the
# input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
# check to see if we are using the *fast* but *less accurate* version
# of selective search

ss.switchToSelectiveSearchFast()
# otherwise we are using the *slower* but *more accurate* version


start = time.time()
rects = ss.process()

# show how along selective search took to run along with the total
# number of returned region proposals
print(rects)

output = image.copy()
for (x, y, w, h) in rects:
        if w*h>minArea:
                #d_object=output[y:y+h,x:x+w]
                #d_object=cv2.resize(d_object, (img_width,img_height),interpolation=cv2.INTER_AREA)
                #d_object=cv2.cvtColor(d_object, cv2.COLOR_BGR2RGB)
                #img_array = tf.expand_dims(d_object, 0)
                #predictions = model.predict(img_array)
                #score = tf.nn.softmax(predictions[0])
                #output = cv2.putText(output, "{} {:.2f}".format(class_names[np.argmax(score)], 100 * np.max(score)), (x,y+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
                color = [random.randint(0, 255) for j in range(0, 3)]
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                
                
cv2.imshow("Output", output)
end = time.time()

print("[INFO] selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))
