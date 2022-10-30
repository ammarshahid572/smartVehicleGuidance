import numpy as np
import cv2
img_height = 120
img_width = 180
file=r"D:\Python\SmartVehicleGuidance\test2.jpg"

img= cv2.imread(file)

binary= np.zeros_like(img)
minArea=5000
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
rects = ss.process()
print(len(rects))
for (x, y, w, h) in rects:
    if w*h>minArea:
        cv2.rectangle(binary, (x, y), (x + w, y + h), (255,255,255), 2)
        
binary= cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
t,binary= cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
binary = cv2.erode(binary, kernel, iterations=2)
#binary = cv2.dilate(binary, kernel, iterations=2)


contours, hierarchies= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i,contour in enumerate(contours):
    if (hierarchies[0][i][3])<=0:
        (x,y,w,h) = cv2.boundingRect(contour)
        if w*h > minArea:
                #d_object=img[y:y+h,x:x+w]
                #d_object=cv2.resize(d_object, (img_width,img_height),interpolation=cv2.INTER_AREA)
                #d_object=cv2.cvtColor(d_object, cv2.COLOR_BGR2RGB)
                #img_array = tf.expand_dims(d_object, 0)
                #predictions = model.predict(img_array)
                #score = tf.nn.softmax(predictions[0])
                cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
                #img = cv2.putText(img, "{} {:.2f}".format(class_names[np.argmax(score)], 100 * np.max(score)), (x,y+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)

cv2.imshow("Img", img)
cv2.imshow("binary",binary) 
