#using CNN with edge detection + contours


import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from b_serial import sendColor
import json
import pathlib
print(tf.__version__)
# colors 0 red, 1 green, 2, yellow, 3, emergency
from keras.models import load_model
model = load_model(r'models/vehicles_light_new.h5')
SerialDump=dict()
import threading

minArea=10000
filepath=r"D:\Python\SmartVehicleGuidance\testImages\test_new\test1.jpg"
img_height = 120
img_width = 180
model.summary()

def nothing(value):
    pass
    
class_names = ['car', 'otherStuff', 'truck']

lane1= [0 , 250]

laneColors=[1,1,1]
colorsChanged=False
ready=True
img=cv2.imread(filepath)

def colorUpdate(SerialDump):
    y=json.dumps(SerialDump)
    sendColor(y)
    time.sleep(1)
    colorsChanged=False
    ready=True

#Uncomment to use Live Video Feed
cap= cv2.VideoCapture(1)
Once=True
t1 = threading.Thread(target=colorUpdate, args=(SerialDump,))
iterations_Dilate=0
iterations_Erode=0



cv2.namedWindow("Settings", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('MinArea', "Settings", 5, 100, nothing)
cv2.createTrackbar('Dilation', "Settings", 0, 5, nothing)
cv2.createTrackbar('Erosion', "Settings", 0, 5,  nothing)
cv2.createTrackbar('Brightness', "Settings", 0, 200,  nothing)
cv2.createTrackbar('Blur', "Settings", 1, 9,  nothing)
cv2.createTrackbar('TopLine', "Settings", 0, 50,  nothing)
cv2.createTrackbar('FirstLane', "Settings", 100, 300,  nothing)


prevColors=[0,0,0]
while Once:
    start=time.time()
    #Uncomment to use a single file
    #img=cv2.imread(filepath)

    #Uncomment to use live video Feed
    _,img=cap.read()
    
    img= cv2.resize(img, (600,600),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix = np.ones(img.shape, dtype = "uint8") * cv2.getTrackbarPos('Brightness','Settings')
    blurIndex=cv2.getTrackbarPos('Blur','Settings')
    if blurIndex>0:
        img = cv2.blur(img,(blurIndex,blurIndex))
    l1Endpoint=cv2.getTrackbarPos('FirstLane','Settings')
    lane1[1]=l1Endpoint 
    y=cv2.getTrackbarPos('TopLine','Settings')
    img= cv2.line(img, (l1Endpoint,500),(l1Endpoint,y), (255,255,255), 5)
    img= cv2.line(img, (0,y),(500,y), (255,255,255), 5)
    kernel = np.array([[-1,-1,-1], 
                   [-1,9,-1], 
                   [-1,-1,-1]])
    #img = cv2.filter2D(img, -1, kernel)
    
    img = cv2.add(img, matrix)
    canny= cv2.Canny(gray, 50, 100)
    kernel = np.ones((3,3), np.uint8)
    kernel2 = np.ones((3,3), np.uint8)
    
    
    dilated = cv2.dilate(canny, kernel2, iterations=cv2.getTrackbarPos('Dilation','Settings'))
    eroded = cv2.erode(dilated, kernel, iterations=cv2.getTrackbarPos('Erosion','Settings'))
    eroded[y+1]=0
    eroded[y]=0
    eroded[y-1]=0
    
    cv2.imshow("binary", eroded)
    contours, hierarchies= cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i,contour in enumerate(contours):
        if (hierarchies[0][i][3])<0:
            (x,y,w,h) = cv2.boundingRect(contour)
            if w*h > (cv2.getTrackbarPos('MinArea','Settings')*1000):
                    d_object=img[y:y+h,x:x+w]
                    d_object=cv2.resize(d_object, (img_width,img_height),interpolation=cv2.INTER_AREA)
                    d_object=cv2.cvtColor(d_object, cv2.COLOR_BGR2RGB)
                    img_array = tf.expand_dims(d_object, 0)
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
                    objectDetected= class_names[np.argmax(score)]
                    
                    img = cv2.putText(img, "{} {:.2f}".format(class_names[np.argmax(score)], 100 * np.max(score)), (x,y+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
                    if objectDetected=="truck":
                        if ((x+w)/2)>lane1[0] and ((x+w)/2)<lane1[1]:
                            print("Truck in wrong lane")
                            laneColors[0]=0
                            colorsChanged=True
                            break
                    elif objectDetected=="car":
                        if ((x+w)/2)>lane1[0] and ((x+w)/2)<lane1[1]:
                            laneColors[0]=1
                            colorsChanged=True
                            
                            
##    if prevColors[0]!= laneColors[0]:
##        colorsChanged=True
##        prevColors[0]= laneColors[0]
                            
              
    SerialDump["l1"]= laneColors[0]
    SerialDump["l2"]= laneColors[1]
    SerialDump["l3"]= laneColors[2]
    
    if colorsChanged:
        try:
            if not t1.is_alive():
                t1.start()
            else:
                print("Thread is already running")
        except:
            print("waiting on thread")
            t1 = threading.Thread(target=colorUpdate, args=(SerialDump,))
            t1.start()
        colorsChanged=False

    end=time.time()
    fps= str(int(1/(end-start)))
    img = cv2.putText(img, fps, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4, cv2.LINE_AA)
    cv2.imshow("Window",img)
    k=cv2.waitKey(1)
    if k & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
             
