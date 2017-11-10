#load 2 files, 3 file
import csv
import numpy as np
import pandas as pd

def done(): 
    print("Done.\n")

def write(data, file, header="Id,Label"):
    with codecs.open(file, 'w') as f:
        f.write("{0}\n".format(header))
        for i,v in enumerate(data):
            f.write("{0},{1}\n".format(i,v))

def load_datasets(train_x, train_y):
    print("Loading data sets...")

    X = np.loadtxt(train_x, delimiter=",")
    Y = np.loadtxt(train_y, dtype=(np.int32))
    print(Y[0])
    
    done()
#     return X, Y

# def data_wrap():
#     X, Y = load_datasets(train_x, train_y)
    training_inputs = [np.reshape(x, (4096, 1)) for x in X]
    training_results = []
    training_results.append(output_classes(y) for y in Y)
    training_data = list(zip(training_inputs, training_results))

    # validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # validation_data = zip(validation_inputs, va_d[1])
    # test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    # test_data = zip(test_inputs, te_d[1])
    # return (training_data, validation_data, test_data)
    return training_data

def output_classes(o): 
    #vectorize result. 40 classes.
    classes = np.zeros((40, 1))
    classes[o] = 1.0
    return classes

if __name__ == '__main__':
    train_x= '/Users/michliu/PycharmProjects/sample_train_x.csv'
    train_y = '/Users/michliu/PycharmProjects/sample_train_y.csv'
    #load_datasets(train_x, train_y)
    data_wrap()
    
