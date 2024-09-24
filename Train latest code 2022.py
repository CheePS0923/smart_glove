import serial
import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv

ser = serial.Serial('COM4', 9600)#timeout = 5
ser.flushInput()
time.sleep(3)
emp = []
val1 = []
DATASET = 200 #number of datapoints/set
FILE = "data_6_0_"
#(eg: startSet = 3, samples 1 & samples 2 output not affected, sample 3 onwards updated to new output)
startSet = 0 #writes the desired output starting from this sample set
data = np.zeros(shape=(DATASET,3))
output = 0
filecount= 0 # eg: put 7 if want data_x_x_8

##startIndex = (startSet*DATASET)-DATASET
##if startIndex < 0:
##    startIndex = 0

dataset_dict = {
    'shape': {
        1: 'Rectangular', 
        2: 'Cylinder', 
        3: 'Pyramid', 
        4: 'Sphere', 
        5: 'Big Sphere', 
        6: 'Dodecahedron', 
    }
}

while True:
    while True: #determine when the button is pressed, send '1' when button pushed
        print("Push button to read data")
        startBit = ser.readline()
        if startBit == b'1\r\n':
            print("Button Pushed!")
            ser.write(b'1')
            break;

    for datapoint in range(DATASET+1):
        ser_bytes = ser.readline()
        #print("Received: " + str(ser_bytes))
        if datapoint != 0: #ignore first read (bad data)
            #decode from bytes to str remove \r\n from end of line, and turn to list
            decode = ser_bytes.decode("utf-8")[:-2].split()
            #['sensorValue=', '142', 'sensorValue1=', '935', 'sensorValue2=', '199']
            count = 0
            for ind, val in enumerate(decode): #arrange values to array (row = trials, col= finger)
                if ind%2: #find odd no.
                    data[datapoint-1][count] = float(val)
                    count+=1
            #data[datapoint-1][5] = 4
    ser.write(b'S')
    print("Stop sentinel sent!")
    filecount+=1
    
    with open(FILE+str(filecount)+'.csv','w') as f: #save data
        np.savetxt(f, data, delimiter = ',',fmt='%i')

##    df = pd.read_csv('Training_data.csv',header=None)
##    totalRow = len(df.index)
##    print(totalRow/DATASET)
##    for row in range(startIndex,totalRow):
##        if (row%DATASET)== 0:
##            df.loc[row,5] = output
            
##    df.to_csv('Training_data.csv', mode='w', index = False, header = False)
            

    plt.plot(data)
    plt.ylabel('bend angle')
    plt.show()