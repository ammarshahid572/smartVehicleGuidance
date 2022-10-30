import cv2
import numpy as np 

def nothing(x):
    pass


testPath=r"D:\Python\SmartVehicleGuidance\data\vehicles\cars\cartest2.png"


itemsParams= [
                ["car" ,[31,64,128], [31,100,100], [41,255,225] , 500],
                ["truck" ,[96,255,66], [86,100,100], [106,255,255] , 2000 ]
            ]


def hsv_selection(frame):
    
    frame= cv2.resize(frame,(120,180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for itemParams in itemsParams:
        
        itemLabel = itemParams[0]
        itemHSV =  np.uint8(itemParams[1])
        itemUL = np.uint8(itemParams[2])
        itemLL = np.uint8(itemParams[3])
        itemArea= itemParams[4]


        mask = cv2.inRange(hsv, itemUL, itemLL)
        mask= cv2.dilate(mask, (3,3), iterations=10)
        whitePixelCount = cv2. countNonZero(mask)
        #cv2.imshow(itemLabel,mask)

        if whitePixelCount>itemArea:
            return(itemLabel)
        
    return "other Objects"

if __name__=="__main__":
    frame= cv2.imread(testPath)
    print(hsv_selection(frame))
    

