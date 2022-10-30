from serial import Serial

import json
import time

#Change Port as per needs
ser = Serial('COM14', 115200, timeout=1)

#color setting function
def sendColor(color):       
    try :
        ser.write(color.encode('utf-8'))
        ser.flush()
    except:
        pass
    
    
if __name__ == '__main__':
   
    laneColors= [0,0,0]
    SerialDump=dict()
    i=0
    while True:
        if i>3:
            i=0
        output=str((i*100)+(i*10)+i)

        i=i+1
        sendColor(output)
        print(output)
        time.sleep(1.3)
