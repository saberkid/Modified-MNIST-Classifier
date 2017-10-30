import numpy as np
import matplotlib.pyplot as plt


class Reader:
    def __init__(self,train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

    def vectorizer(self, line):
        vector = []
        for item in line.strip().split(' '):
            vector.append(int(item))
        return vector

    def read_train(self):
        X_train = []
        y_train = []
        with open(self.train_file, "r") as train_reader:
            while 1:
                line = train_reader.readline()
                if not line:
                    break
                (label, line) = line.split(",")[0], line.split(",")[1]
                line_vector = self.vectorizer(line)
                X_train.append(line_vector)
                y_train.append(int(label))
                # X_train, y_train = np.asarray(X_train), np.asarray(y_train)
                # x = X_train[0].reshape(64, 64)  # reshape
                # y = y_train[0]
                # print x
                # print y
                # plt.imshow(x)
                # plt.show()
                # break
        X_train, y_train = np.asarray(X_train), np.asarray(y_train)
        return X_train, y_train

    def read_val(self):
        X_val = []
        y_val = []
        with open(self.test_file, "r") as test_reader:
            while 1:
                line = test_reader.readline()
                if not line:
                    break
                (label, line) = line.split(",")[0], line.split(",")[1]
                line_vector = self.vectorizer(line)
                X_val.append(line_vector)
                y_val.append(int(label))
        X_val, y_val = np.asarray(X_val), np.asarray(y_val)
        return X_val, y_val

    def read_test(self):
        X_test = []
        with open(self.test_file, "r") as test_reader:
            while 1:
                line = test_reader.readline()
                if not line:
                    break
                line = line.replace(',', '').replace('\n', '')
                line_vector = self.vectorizer(line)
                X_test.append(line_vector)
        X_test= np.asarray(X_test)
        return X_test
