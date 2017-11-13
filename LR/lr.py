# Logistic Regression

import scipy.misc as dfg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import io
import math
import csv
import operator
import random
import time as time
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

x = np.loadtxt("../data/train_x.csv", delimiter=",") 
test = np.loadtxt("../data/test_x.csv", delimiter=",") 
y = np.loadtxt("../data/train_y.csv", delimiter=",") 

print y
print x
print test

# Perform classification 
logistic = LogisticRegression()
t0 = time.time()
logistic.fit(x,y)
t1 = time.time()
prediction = logistic.predict(test)
t2 = time.time()

time_lr_train = t1-t0
time_lr_predict = t2-t1


print("Results for Logistic Regression")
print("Training time: %fs; Prediction time: %fs" % (time_lr_train, time_lr_predict))    

seq_num = xrange(len(test))
with open('lr_predictions.csv','w+') as predict_writer:
    predict_writer.writelines('Id,Label\n')
    for test_num in seq_num:          
        label = prediction[test_num]
        predict_writer.writelines(str(test_num+1) + ',' + str(label) + '\n')
