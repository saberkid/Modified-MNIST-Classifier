import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential


def vectorizer( line):
    vector = []
    for item in line.strip().split(' '):
        vector.append(int(item))
    return vector
model = Sequential()
model.add(Convolution2D(64,    # number of filter layers
                        [3,3],    # x dimension of kernel
                        input_shape=[64, 64, 1]))
#model.add(MaxPooling2D(pool_size=(2, 2)))
with open('../data/test_x.csv_nbg', "r") as reader :
    while 1:
        line = reader.readline()
        if not line:
            break
        line = line.replace(',', '').replace('\n', '').replace('\t', '')
        line_vector = vectorizer(line)
        X_test = np.asarray(line_vector)
        plt.imshow(X_test.reshape([64, 64]), cmap=plt.get_cmap('gray'))
        plt.show()
        x = X_test.reshape([64, 64, 1])  # reshape
        x = np.expand_dims(x, axis=0)
        conved = model.predict(x)
        conved = np.squeeze(conved, axis=0)
        for i in range(64):
            mat = conved[:, :, i]
            plt.imshow(mat, cmap = plt.get_cmap('gray'))
            plt.show()
        break
