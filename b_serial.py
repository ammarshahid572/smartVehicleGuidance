from serial import Serial
import time

ser = Serial('COM7', 9600, timeout=1)

def sendColor(color):
    ser.flush()
    ser.write(color.encode('utf-8'))
    
if __name__ == '__main__':
   
    ser.flush()
    sendColor("red")
    line = ser.readline().decode('utf-8').rstrip()
    print(line)
