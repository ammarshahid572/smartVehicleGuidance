import cv2


cap=cv2.VideoCapture(1)

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
 
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
 
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

while True:
    ret, img= cap.read()
    img = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)

    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
     
    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)
     
        text = '%s: %.2f' % (classes[classId[0]], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=2)


    cv2.imshow('Image', img)
    k=cv2.waitKey(30)
    if k &0xff==27:
        break

cap.release()
cv2.destroyAllWindows()
