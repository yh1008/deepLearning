
# coding: utf-8

# This script is used to grid search parameters on GPU
# 

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras import backend as K
from keras.layers import Dropout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# load data
from keras.datasets import mnist


def mnist_data_split():
    r"""return processed train and test data,
    the actual process takes place here includes:
    1. reshape to nofrows, 28, 28, 1
    2. shuffle
    3. dummify labels
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
    # the data, shuffled and split between train and test sets
    
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    
    print("after process: X train shape: {}, X test shape: {}, y train shape: {}, y test shape: {}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))
    return  x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = mnist_data_split()


print ("inputshape is : {}".format(x_train.shape[1]))

# vanilabase line
def train_mnist_nnet( x_train, x_test, y_train, y_test):
    r"""
    Returns
    -------
    hisotry_callback : <class 'keras.callbacks.History'>
        used for retrospective examiniation
    """
    
    num_class = 10
    epochs = 12
    batch_size = 128
    input_shape = (784, ) #(784, )
    model = Sequential()
    model.add(Dense(32, input_shape=input_shape, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])
    
    history_callback = model.fit(
                  x_train, 
                  y_train, 
                  verbose=0, 
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  epochs=epochs
              )
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))
    #print(model.summary())

    return history_callback

# callback = train_mnist_nnet(x_train, x_test, y_train, y_test)

# grid search parameters (number of epoches, hidden units) without dropout
def make_mnist_nnet(hidden_size=16):
    r"""
    Returns
    -------
    model : model itself
    """
    
    num_class = 10
    input_shape = (784, ) #(784, )
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=input_shape, activation="relu"))
    model.add(Dense(int(hidden_size/2), activation="relu"))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])


    return model



clf = KerasClassifier(make_mnist_nnet)
param_grid = {'epochs': [12, 32, 64], 'hidden_size': [32, 64], 'batch_size':[128]}
grid = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid.fit(x_train, y_train.astype('int32'))

print ("best params: {}".format(grid.best_params_))
print ("mean test score: {} ".format(grid.cv_results_['mean_test_score']))
print ("mean train score: {} ".format(grid.cv_results_['mean_train_score']))


'''
# #### Add drop out
def train_mnist_dropout_nnet( x_train, x_test, y_train, y_test):
    r"""
    Returns
    -------
    hisotry_callback : <class 'keras.callbacks.History'>
        used for retrospective examiniation
    """
    
    num_class = 10
    epochs = 64 #32
    batch_size = 128
    input_shape = (784, ) #(784, )
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])
    
    history_callback = model.fit(
                  x_train, 
                  y_train, 
                  verbose=0, 
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  epochs=epochs
              )
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))
    #print(model.summary())

    return history_callback
#callback1 = train_mnist_dropout_nnet( x_train, x_test, y_train, y_test)

'''
# grid search dropout rate
def make_mnist_nnet_dropout(dropout_rate=0.2):
    r"""
    Returns
    -------
    model : model itself
    """
    num_class = 10
    hidden_size = 64
    input_shape = (784, ) #(784, )
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=input_shape, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(hidden_size/2), activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])


    return model

clf1 = KerasClassifier(make_mnist_nnet_dropout)
param_grid1 = {'epochs': [64], 'dropout_rate': [0.1, 0.2, 0.3], 'batch_size':[128]}
grid1 = GridSearchCV(clf1, param_grid=param_grid1, cv=5)
grid1.fit(x_train, y_train.astype('int32'))

print ("best params: {}".format(grid1.best_params_))
print ("mean test score: {} ".format(grid1.cv_results_['mean_test_score']))
print ("mean train score: {} ".format(grid1.cv_results_['mean_train_score']))


def train_mnist_nnet_advanced( x_train, x_test, y_train, y_test, add_dropout=False, hidden_size=64, epochs=32, dropout_rate=0.1):
    r"""
    Returns
    -------
    hisotry_callback : <class 'keras.callbacks.History'>
        used for retrospective examiniation
    """
    
    num_class = 10
    batch_size = 128
    input_shape = (784, ) #(784, )
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=input_shape, activation="relu"))
    if add_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_size/2, activation="relu"))
    if add_dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])
    
    history_callback = model.fit(
                  x_train, 
                  y_train, 
                  verbose=0, 
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  epochs=epochs
              )
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))
    #print(model.summary())
    return history_callback

c1 = train_mnist_nnet_advanced(x_train, x_test, y_train, y_test, hidden_size=64, epochs=64 )
c2 = train_mnist_nnet_advanced(x_train, x_test, y_train, y_test, add_dropout=True, hidden_size=64, epochs=64 )
hist_c1 = pd.DataFrame(c1.history)
hist_c2 = pd.DataFrame(c2.history)
hist_c2.rename(columns=lambda x: x + "_dropout", inplace=True)
hist_c2[['acc_dropout', 'val_acc_dropout']].plot()
hist_c1[['acc', 'val_acc']].plot(ax=plt.gca(), linestyle='--')
plt.title("Dropout vs. NoDropout Learning Curve")
plt.ylim(.8, 1)
plt.savefig('learning_curve_compare.png')
