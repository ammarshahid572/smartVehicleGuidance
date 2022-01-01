import cv2
import sys
import numpy as np
from time import sleep
#from b_serial import sendColor
casc1Path = "truck.xml"
casc2Path = "cascade.xml"


trucksCascade = cv2.CascadeClassifier(casc1Path)
carsCascade = cv2.CascadeClassifier(casc2Path)

video_capture = cv2.VideoCapture(1)
img_height = 500
img_width = 500
dim = (img_width, img_height)
linecolor=(0,255,0)
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    trucks = trucksCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    cars = carsCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in cars:
        print("car detected")
        face = np.ones((w,h,3), np.uint8)
        centerx=x+w/2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in trucks:
        face = np.ones((w,h,3), np.uint8)
        centerx=x+w/2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
