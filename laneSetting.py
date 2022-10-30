import time
import json
from b_Serial import sendColor
laneColors= [0,0,0]
SerialDump=dict()
i=0
while True:
    if i>3:
        i=0
    SerialDump["l1"]= i
    SerialDump["l2"]= i
    SerialDump["l3"]= i

    i=i+1

    y=json.dumps(SerialDump)
    print(y)
    time.sleep(2)
