import numpy as np
import matplotlib.pyplot as plt


def vectorizer( line):
    vector = []
    for item in line.strip().split(' '):
        vector.append(int(item))
    return vector

with open('../data/test_x.csv_nbg', "r") as reader:
    while 1:
        line = reader.readline()
        if not line:
            break
        line = line.replace(',', '').replace('\n', '').replace('\t', '')
        line_vector = vectorizer(line)
        X_test = np.asarray(line_vector)
        x = X_test.reshape(64, 64)  # reshape

        print x
        plt.imshow(x, cmap = plt.get_cmap('gray'))
        plt.show()
