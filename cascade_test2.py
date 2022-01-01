import cv2
#from b_serial import sendColor
################################################################
path1 = 'truck.xml'
path2 = 'car.xml'         # PATH OF THE CASCADE
cameraNo = 1                    # CAMERA NUMBER
label1 = 'truck'       # OBJECT NAME TO DISPLAY
label2 = 'car'  
frameWidth= 640                     # DISPLAY WIDTH
frameHeight = 480                  # DISPLAY HEIGHT
color1= (0,0,255)
color2= (0,255,0)

#################################################################


cap = cv2.VideoCapture(cameraNo)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Scale","Result",400,1000,empty)
cv2.createTrackbar("Neig","Result",8,50,empty)
cv2.createTrackbar("Min Area","Result",0,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)


cascade1 = cv2.CascadeClassifier(path1)
cascade2 = cv2.CascadeClassifier(path2)
prevDetect='none'
detected ='none'
while True:
    # SET CAMERA BRIGHTNESS FROM TRACKBAR VALUE
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)
    # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # DETECT THE OBJECT USING THE CASCADE
    scaleVal =1 + (cv2.getTrackbarPos("Scale", "Result") /1000)
    neig=cv2.getTrackbarPos("Neig", "Result")
    
    trucks = cascade1.detectMultiScale(gray,scaleVal, neig)
    cars = cascade2.detectMultiScale(gray,scaleVal, neig)
    # DISPLAY THE DETECTED OBJECTS
    for (x,y,w,h) in trucks:
        area = w*h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area >minArea:
            detected='truck'
            print('Detected a truck')
            cv2.rectangle(img,(x,y),(x+w,y+h),color1,3)
            cv2.putText(img,label1,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color1,2)
            roi_color = img[y:y+h, x:x+w]

    for (x,y,w,h) in cars:
        area = w*h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area >minArea:
            print('detected a car')
            detected='car'
            cv2.rectangle(img,(x,y),(x+w,y+h),color2,3)
            cv2.putText(img,label2,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color2,2)
            roi_color = img[y:y+h, x:x+w]
    if prevDetect!= detected:
        prevDetect= detected
        if detected=='truck':
            print("Senging Red")
            #sendColor('red')
        else:
            print("Sending Green")
            #sendColor('green')
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()
