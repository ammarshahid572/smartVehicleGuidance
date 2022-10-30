#using CNN with edge detection + contours


import tensorflow as tf
import numpy as np
import cv2
import time
from b_serial import sendColor
import json
import pathlib

from Rcnn import class_extraction
from hsv_selection import hsv_selection

import threading


minArea=10000
filepath=r"testImages\tests\video (1).mp4"
img_height = 120
img_width = 180

SerialDump="000"

frame_width=600
frame_height=600
def nothing(value):
    pass
    


lane1= [0 , 250]

laneColors=[1,1,1]
colorsChanged=False
ready=True
img=cv2.imread(filepath)

def colorUpdate(SerialDump):
    sendColor(SerialDump)
    time.sleep(1.3)
    colorsChanged=False
    ready=True

#Uncomment to use Live Video Feed
cap= cv2.VideoCapture(1)
Once=True
t1 = threading.Thread(target=colorUpdate, args=(SerialDump,))


cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
cv2.createTrackbar('MinArea', "Settings", 5, 30, nothing)
cv2.createTrackbar('MaxArea', "Settings", 20, 40, nothing)
cv2.createTrackbar('Dilation', "Settings", 0, 5, nothing)
cv2.createTrackbar('Erosion', "Settings", 0, 5,  nothing)
cv2.createTrackbar('Brightness', "Settings", 0, 50,  nothing)
cv2.createTrackbar('Blur', "Settings", 1, 9,  nothing)
cv2.createTrackbar('TopLine', "Settings", 0, 25,  nothing)
cv2.createTrackbar('BotLine', "Settings", 0, 25,  nothing)
cv2.createTrackbar('FirstLane', "Settings", 10, 30,  nothing)


prevColors=[0,0,0]

useHsvBoost=False
while Once:
    start=time.time()
    #Uncomment to use a single file
    #img=cv2.imread(filepath)

    #Uncomment to use live video Feed
    _,img=cap.read()
    
    img= cv2.resize(img, (frame_width, frame_height),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix = np.ones(img.shape, dtype = "uint8") * cv2.getTrackbarPos('Brightness','Settings')
    blurIndex=cv2.getTrackbarPos('Blur','Settings')
    if blurIndex>0:
        img = cv2.blur(img,(blurIndex,blurIndex))
    l1Endpoint=(cv2.getTrackbarPos('FirstLane','Settings')*10)
    lane1[1]=l1Endpoint 
    topLineY=(cv2.getTrackbarPos('TopLine','Settings')*2)
    bottomLineY=(cv2.getTrackbarPos('BotLine','Settings')*2)
    
    img= cv2.line(img, (0,topLineY),(500,topLineY), (255,255,255), 5)
    img= cv2.line(img, (0,frame_height-bottomLineY),(500,frame_height-bottomLineY), (255,255,255), 5)
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
    eroded[topLineY+1]=0
    eroded[topLineY]=0
    eroded[topLineY-1]=0
    
    cv2.imshow("binary", eroded)
    contours, hierarchies= cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i,contour in enumerate(contours):
        if (hierarchies[0][i][3])<0:
            (x,y,w,h) = cv2.boundingRect(contour)
            if w*h > (cv2.getTrackbarPos('MinArea','Settings')*1000) and w*h < (cv2.getTrackbarPos('MaxArea','Settings')*1000) and w<300 and x>50 and x+w<frame_width-10 and y>10 and y+h<(frame_height-bottomLineY):
                    d_object=img[y:y+h,x:x+w]
                    objectDetected="invalid"
                    confi=0
                    if useHsvBoost:
                        objectDetected, confi=hsv_selection(d_object)
                    else:
                        objectDetected, confi= class_extraction(d_object)
                    #img = cv2.putText(img, "{} {:.2f}".format(objectDetected,confi), (x,y+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
                    if confi>60  and h/w<3:
                        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
                        img = cv2.putText(img, "{} {:.2f}".format(objectDetected,confi), (x,y+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
                        if objectDetected=="ambulance":
                            if ((x+w)/2>lane1[0] and (x+w/2)<lane1[1]):
                                laneColors[0]=2
                            elif ((x+w)/2>lane1[1]) and (x+w/2)<(lane1[1]*2):
                                laneColors[0]=2
                                laneColors[1]=2
                            else:
                                laneColors[1]=2
                                laneColors[2]=2
                            colorsChanged=True
                            break
                        elif objectDetected=="truck" and ((x+w)/2)>lane1[0] and ((x+w)/2)<lane1[1]:
                                print("Truck in wrong lane")
                                laneColors[0]=3
                                colorsChanged=True
                                break
                        elif objectDetected=="car":
                             
                            laneColors[0]=1
                            laneColors[1]=1
                            laneColors[2]=1
                            colorsChanged=True
                            
              
    SerialDump=str(laneColors[0]*100+ laneColors[1]*10+ laneColors[2])
    
    if colorsChanged:
        print(SerialDump)
        try:
            if not t1.is_alive():
                t1.start()
            else:
                nothing(0)
                print("Thread is already running")
        except:
            print("waiting on thread")
            t1 = threading.Thread(target=colorUpdate, args=(SerialDump,))
            t1.start()
        colorsChanged=False

    end=time.time()
    fps= str(int(1/(end-start)))
    img= cv2.line(img, (l1Endpoint,500),(l1Endpoint,topLineY), (255,255,255), 5)
    
    img = cv2.putText(img, fps, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4, cv2.LINE_AA)
    cv2.imshow("Window",img)
    k=cv2.waitKey(1)
    if k & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
             
