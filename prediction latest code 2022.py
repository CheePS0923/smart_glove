#import tensorflow as tf
import serial
import time
##from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import os
from os.path import isfile, join
import glob
import matplotlib.image as mpimg
from sklearn.datasets import make_classification
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.neural_network
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import cm
from sklearn.metrics import confusion_matrix, accuracy_score

DATASET = 200
dataset_dict = {
    'shape': {
        1: "C:\\Users\\HOME\\OneDrive - Universiti Tunku Abdul Rahman\\Desktop\\latest_code_2022\\shape\\1.png", 
        2: "C:\\Users\\HOME\\OneDrive - Universiti Tunku Abdul Rahman\\Desktop\\latest_code_2022\\shape\\2.png",
        3: "C:\\Users\\HOME\\OneDrive - Universiti Tunku Abdul Rahman\\Desktop\\latest_code_2022\\shape\\3.png",
        4: "C:\\Users\\HOME\\OneDrive - Universiti Tunku Abdul Rahman\\Desktop\\latest_code_2022\\shape\\4.png",
        5: "C:\\Users\\HOME\\OneDrive - Universiti Tunku Abdul Rahman\\Desktop\\latest_code_2022\\shape\\5.png",
        6: "C:\\Users\\HOME\\OneDrive - Universiti Tunku Abdul Rahman\\Desktop\\latest_code_2022\\shape\\6.png",
    }
}


#serial comm
# =============================================================================
# ser = serial.Serial('COM13', 9600)#timeout = 5
# ser.flushInput()
# time.sleep(3)
# =============================================================================

fileList = glob.glob("1_DATA COMPILED\*.csv")
x = []
y = []
#read train data
for fileName in fileList:
    with open(fileName, 'r') as f:
        data = f.readlines()
        #convert to float and list
        data = [list(map(float, i.strip().split(','))) for i in data]
    x.append(data)
        
    #get the label from filename and append it to a list
    y.append(int(fileName.split('_')[2]))

x = np.array(x)
y = np.array(y)

##print(x.shape)
x = x.reshape((1200, 200*3))

#normalize X
##xmax, xmin = x.max(), x.min()
##x = (x-xmin)/(xmax-xmin)

#80% train(160 samples from each object), 20% test(40 samples from each object)
SEED = 108
# X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state = SEED,stratify = y)
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state = SEED,stratify = y)

#Gradient Boosting Classifier
# clf = GradientBoostingClassifier(n_estimators=20)

#Extra Trees Classifier
# clf = ExtraTreesClassifier(n_estimators=10)

#Decision Tree Classifier
# clf = DecisionTreeClassifier()

#K-Naerest Neighbors Classifier
# clf = KNeighborsClassifier(n_neighbors=5)

#Naive Bayes Classifier
# clf = GaussianNB()

#Random Forest Classifier
# clf = RandomForestClassifier(n_estimators = 10, random_state = 30)

#SVM Classifier
clf = svm.SVC(kernel = 'rbf')#, probability = True)

#Logistic Regression Classifier
# clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')

#fitting data to classfier
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

#prediction accuracy
print("accuracy", metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred, average = None))
print("Recall:",metrics.recall_score(Y_test, y_pred, average = None))

# Calculate confusion matrix and accuracy
cm_matrix = confusion_matrix(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create the heatmap with the confusion matrix
cax = ax.matshow(cm_matrix, cmap=cm.Blues)

# Add a colorbar
plt.colorbar(cax)

# Annotate each cell with the raw value and percentage
for (i, j), value in np.ndenumerate(cm_matrix):
    percentage = value / np.sum(cm_matrix) * 100  # Calculate percentage for each element
    ax.text(j, i, f'{value}\n({percentage:.2f}%)', ha='center', va='center', color='black')

# Add labels, title, and accuracy annotation
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')

plt.show()

# =============================================================================
# cmap = cm.get_cmap('Blues')  # You can change 'Blues' to other colormaps
# 
# # plot_confusion_matrix(clf, X_test, Y_test)
# cm_display = ConfusionMatrixDisplay.from_estimator(clf, X_test, Y_test, cmap=cmap)
# plt.show()
# =============================================================================

# =============================================================================
# def read_real_time_data(DATASET): #read data from glove
#     data = np.zeros(shape=(DATASET,3))
#     while True: #determine when the button is pressed, send '1' when button pushed
#         print("Push button to read data")
#         startBit = ser.readline()
#         if startBit == b'1\r\n':
#             print("Button Pushed!")
#             ser.write(b'1')
#             break;
#     
#     for datapoint in range(DATASET+1):
#         ser_bytes = ser.readline()
#         #print("Received: " + str(ser_bytes))
#         if datapoint != 0: #ignore first read (bad data)
#             #decode from bytes to str remove \r\n from end of line, and turn to list
#             decode = ser_bytes.decode("utf-8")[:-2].split()
#             #['sensorValue=', '142', 'sensorValue1=', '935', 'sensorValue2=', '199']
#             count = 0
#             for ind, val in enumerate(decode): #arrange values to array (row = trials, col= finger)
#                 if ind%2: #find odd no.
#                     data[datapoint-1][count] = float(val)
#                     count+=1
#     ser.write(b'S')
#     print("Stop sentinel sent!")
# 
# ##    print(data)
#     ax1.plot(data)
#     ax1.set_title("Raw Data")
#     data = np.array(data)
#     return data
# =============================================================================
# =============================================================================
# 
# while True:
#     f, (ax1, ax2) = plt.subplots(1,2)
#     x_rt = read_real_time_data(DATASET)
#     x_rt = x_rt.reshape((1, 200*3))
#     
#     y_rt = clf.predict(x_rt)
#     print(y_rt)
# ##    y_prob = clf.predict_proba(x_rt)
# ##    print(y_prob)
# ##    print(y_realtime)
#     imgLoc = dataset_dict['shape'][int(y_rt)]
#     print(imgLoc)
#     img = mpimg.imread(imgLoc)
#     ax2.imshow(img)
#     ax2.set_title("Predicted Object")
#     plt.show()
# =============================================================================
