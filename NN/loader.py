import numpy as np

class Reader:
    def __init__(self, train_file, test_file=None):
        self.train_file = train_file
        self.test_file = test_file
        self.num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30,
                         32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
        self.mapping = {}
        for index, item in enumerate(self.num_list):
            onehot = [0] * 40
            onehot[index] = 1
            self.mapping[item] = onehot

    def vectorizer(self, line):
        vector = []
        for item in line.strip().split(' '):
            vector.append(int(item)/255)
        return vector

    def read_train(self):
        with open(self.train_file, "r") as test_reader:
            cnt = 0
            x = []
            y = []
            while 1:
                # if cnt >= 128:
                #     break
                # cnt += 1
                line = test_reader.readline()
                if not line:
                    break
                (label, line) = line.split(",")[0], line.split(",")[1]
                line_vector = self.vectorizer(line)
                x.append(line_vector)
                y.append(self.mapping[int(label)])
        return (np.asarray(x), np.asarray(y))

    def read_test(self):
        test_data = []
        with open(self.test_file, "r") as test_reader:
            x = []
            while 1:
                line = test_reader.readline()
                if not line:
                    break
                line = line.replace(',', '').replace('\n', '')
                line_vector = self.vectorizer(line)
                x.append(line_vector)
        return x

    def read_val(self):
        x = []
        y = []
        with open(self.test_file, "r") as test_reader:
            cnt = 0
            while 1:
                # if cnt >= 100:
                #     break
                # cnt += 1
                line = test_reader.readline()
                if not line:
                    break
                (label, line) = line.split(",")[0], line.split(",")[1]
                line_vector = self.vectorizer(line)
                x.append(line_vector)
                y.append(self.mapping[int(label)])
        return (np.asarray(x), np.asarray(y))

