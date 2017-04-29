# #### Test accuracy: 0.7554

import keras
from sklearn.cross_validation import train_test_split
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Activation
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imread
import scipy.io as sio
import re
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.linear_model import LogisticRegressionCV
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
import os

path_to_pets = "../pets/" # manually code the path, change it when needed

# build the VGG16 network
# pip install h5py
model = VGG16(include_top=False,
                           weights='imagenet')

cwd = os.getcwd()
parent_path = os.path.split(os.getcwd())[0]
print ("parent path is {}".format(parent_path))


# from keras.preprocessing import image
dir_list = os.listdir(path_to_pets)
dir_list = [f for f in os.listdir(path_to_pets) if re.match(r'[^\\]+\.jpg', f)]

print ("there are {} images in pets directory".format(len(dir_list)))


images_list = [image.load_img(path_to_pets + f, target_size=(224, 224))
                 for f in dir_list]

print(len(images_list))

X = np.array([image.img_to_array(img) for img in images_list])

print ("X.shape: {}".format(X.shape))

# from keras.applications.vgg16 import preprocess_input
X_pre = preprocess_input(X)
features = model.predict(X_pre)

print("features.shape: {}".format(features.shape))

features_ = features.reshape(7390, -1)
print("features_.shape: {}".format(features_.shape))
np.savetxt('features_.out', features_)   # 

# create labels e.g. basset_hound

labels = []
for f in dir_list:
    jpg_removed = f.split(".")[0]
    labels.append(jpg_removed.rsplit("_",1)[0])

np.savetxt('features_.out', features_) 


from sklearn import preprocessing
features_scaled = preprocessing.scale(features_)


sets = list(set(labels))
from collections import defaultdict
label_map_int = defaultdict(int)
for i in range(37):
    label_map_int[sets[i]] = i

labels_ = [label_map_int[l] for l in labels]

labels_dummy = keras.utils.to_categorical(labels_)


X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_dummy, stratify=labels)

print (len(X_train), len(y_train), len(X_test), len(y_test))
print(X_train[0].shape)


# model selection using grid search

def make_nnet(hidden_size=16):
    r"""
    Returns
    -------
    model : model itself
    """
    
    num_class = 37
    input_shape = (25088,)
    
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=input_shape, activation="relu"))
    #model.add(Dense(hidden_size/2, activation="relu"))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])

    return  model

clf = KerasClassifier(make_nnet)
param_grid = {'epochs': [16,32], 'hidden_size': [16,32], 'batch_size':[128]}
grid = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print ("best params: {}".format(grid.best_params_))
#print ("scores: {} ".format(grid.grid_scores_))

# The best parameters are 128 batch size, 32 epoches, and 32 hidden layer, which achieves 0.7341 on the training data

def train_nnet_shallow( x_train, x_test, y_train, y_test):
    r"""
    Returns
    -------
    hisotry_callback : <class 'keras.callbacks.History'>
        used for retrospective examiniation
    """
    
    num_class = 37
    epochs = 32
    batch_size = 128
    input_shape = (25088, ) 
    model = Sequential()
    model.add(Dense(32, input_shape=input_shape, activation="relu"))
    #model.add(Dense(16, activation="relu"))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])
    
    history_callback = model.fit(
                  x_train, 
                  y_train, 
                  verbose=1, 
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  epochs=epochs
              )
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))
    #print(model.summary())

    return history_callback

callback = train_nnet_shallow(X_train, X_test, y_train, y_test)

