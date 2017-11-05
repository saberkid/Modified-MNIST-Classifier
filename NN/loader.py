import numpy as np
import matplotlib.pyplot as plt

class Reader:
    def __init__(self,train_file, label_file): #, test_file = None
        self.train_file = train_file
        #self.test_file = test_file
        self.label_file = label_file
        self.num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
        self.mapping = {}
        for index,item in enumerate(self.num_list):
            onehot = [0]*40
            onehot[index] = 1
            self.mapping[item] = onehot

    def vectorizer(self, line):
        vector = []
        for item in line.strip().split(' '):
            vector.append(int(item))
        return vector

    def read_train(self):
        X = np.loadtxt(self.train_file, delimiter=",") # load from text 
        y = np.loadtxt(self.label_file, delimiter=",") 
        
        #X_train = []
        #y_train = []
        training_data = zip(X,y)

            #line_vector = self.vectorizer(line)
            #X_train.append(line)
            #y_train.append(self.mapping[int(label)])
        return training_data
        #X_train, y_train = np.asarray(X_train), np.asarray(y_train)
        #return X_train, y_train

    # def read_val(self):
    #     X_val = []
    #     y_val = []
    #     with open(self.test_file, "r") as test_reader:
    #         while 1:
    #             line = test_reader.readline()
    #             if not line:
    #                 break
    #             (label, line) = line.split(",")[0], line.split(",")[1]
    #             line_vector = self.vectorizer(line)
    #             X_val.append(line_vector)
    #             y_val.append(self.mapping[int(label)])
    #     X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    #     return X_val, y_val

    # def read_test(self):
    #     X_test = []
    #     with open(self.test_file, "r") as test_reader:
    #         while 1:
    #             line = test_reader.readline()
    #             if not line:
    #                 break
    #             line = line.replace(',', '').replace('\n', '')
    #             line_vector = self.vectorizer(line)
    #             X_test.append(line_vector)
    #     X_test= np.asarray(X_test)
    #     return X_test
