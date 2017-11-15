# SVM

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
from sklearn import svm
from sklearn.metrics import classification_report

x = np.loadtxt("../data/train_x.csv", delimiter=",") 
test = np.loadtxt("../data/test_x.csv", delimiter=",") 
y = np.loadtxt("../data/train_y.csv", delimiter=",") 

print y
print x
print test

# Perform classification 
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(x, y)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))    

seq_num = xrange(len(test))
with open('svm_predictions.csv','w+') as predict_writer:
    predict_writer.writelines('Id,Label\n')
    for test_num in seq_num:          
        label = prediction_liblinear[test_num]
        predict_writer.writelines(str(test_num+1) + ',' + str(label) + '\n')
