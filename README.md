# Modified MNIST Classification
COMP 551 pj3
Team CAM.
overleaf:https://www.overleaf.com/11915340sdswysdbtgzs#/45169305/

## Getting Started
Required packages: [keras 2.0.6](https://keras.io/), [numpy 1.12.1](http://www.numpy.org/), [sklearn](http://scikit-learn.org/stable/)

## Logistic Regression and SVM baseline classifier
In repective folder (LR, SVM), run:
`
python lr.py
python svm.py
`
## Neural Networks from scratch by Python.
Under MLP folder, run:
·
python main.py
·
By default parameter, it would launch a cross-validating training for a two-layer MLP with hidden neurons in range (64, 1024, 64).


## CNN
Under CNN folder, run:
·
python cnn_mnist.py
·

## utils
### imager.py
Visualize image after each layer.
### divider.py
divide traning set into train and validation set. (0.8/0.2 by default)
