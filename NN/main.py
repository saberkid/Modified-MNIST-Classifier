# coding: utf-8
import random
from loader import Reader
from mlp import Mlp
import pickle
import numpy as np


def train(model, (x_train, y_train), (x_test, y_test), learning_rate=0.1, epochs=30, batch_size=32):
    train_size = x_train.shape[0]
    print("Training starts")
    for i in range(epochs):
        x_batches = [
            x_train[k:k + batch_size]
            for k in range(0, train_size, batch_size)]
        y_batches = [
            y_train[k:k + batch_size]
            for k in range(0, train_size, batch_size)]

        for x_batch, y_batch in zip(x_batches, y_batches):
            grad = model.gradient(x_batch, y_batch)
            # update weights and biases
            for key in ('W1', 'b1', 'W2', 'b2'):
                model.params[key] -= learning_rate * grad[key]

        loss = model.loss(x_train, y_train)
        train_loss_list.append(loss)

        train_acc = model.accuracy(x_train, y_train)
        test_acc = model.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


def cross_val(x_tr, y_tr, k=5, validation_flag='hidden'): # validation_flag: 'hidden' or 'lr'
    global train_acc_avg, test_acc_avg, train_acc_list, test_acc_list, train_loss_list
    if validation_flag=='hidden':
        for hiddens in range(64, 1024, 64):
            for i in range(k):
                x_train = list(x_tr)
                y_train = list(y_tr)
                n_sample = len(x_train)
                n_val = n_sample / k
                x_test = x_train[i * n_val: (i + 1) * n_val]
                del x_train[i * n_val: (i + 1) * n_val]
                y_test = y_train[i * n_val: (i + 1) * n_val]
                del y_train[i * n_val: (i + 1) * n_val]
                (x_train, y_train, x_test, y_test) = map(np.asarray, (x_train, y_train, x_test, y_test))

                network = Mlp(input_size=4096, hidden_size=hiddens, output_size=40)
                train(network, (x_train, y_train), (x_test, y_test))
                train_acc_avg.append(train_acc_list[-1])
                test_acc_avg.append(test_acc_list[-1])
                train_acc_list = []
                test_acc_list = []
                train_loss_list = []

            train_avg = reduce(lambda x, y: x + y, train_acc_avg) / len(train_acc_avg)
            test_avg = reduce(lambda x, y: x + y, test_acc_avg) / len(test_acc_avg)
            print("average train acc: %s"%str(train_avg))
            print("average test acc: %s"%str(test_avg))
            train_acc_avg = []
            test_acc_avg = []
         #pickle.dump((train_acc_list, test_acc_list, train_loss_list), open("history_%d"%(hiddens/64),'w'))

    else:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for i in range(k):
                x_train = list(x_tr)
                y_train = list(y_tr)
                n_sample = len(x_train)
                n_val = n_sample / k
                x_test = x_train[i * n_val: (i + 1) * n_val]
                del x_train[i * n_val: (i + 1) * n_val]
                y_test = y_train[i * n_val: (i + 1) * n_val]
                del y_train[i * n_val: (i + 1) * n_val]
                (x_train, y_train, x_test, y_test) = map(np.asarray, (x_train, y_train, x_test, y_test))

                network = Mlp(input_size=4096, hidden_size=256, output_size=40)
                train(network, (x_train, y_train), (x_test, y_test), lr)
                train_acc_avg.append(train_acc_list[-1])
                test_acc_avg.append(test_acc_list[-1])
                train_acc_list = []
                test_acc_list = []
                train_loss_list = []

            train_avg = reduce(lambda x, y: x + y, train_acc_avg) / len(train_acc_avg)
            test_avg = reduce(lambda x, y: x + y, test_acc_avg) / len(test_acc_avg)
            print("average train acc: %s" % str(train_avg))
            print("average test acc: %s" % str(test_avg))
            train_acc_avg = []
            test_acc_avg = []

if __name__ == '__main__':
    train_file = "../data/train_labeled_nbg"
    # test_file = "../data/"
    reader = Reader(train_file)
    x_train, y_train = reader.read_train()

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_acc_avg = []
    test_acc_avg = []

    cross_val(x_train, y_train, k=5, validation_flag='hidden')
