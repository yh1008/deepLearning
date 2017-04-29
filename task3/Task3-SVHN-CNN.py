# task 3 yh2901
import keras
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pandas as pd
#import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.io as sio


def svhn_data_split():
    r"""return processed train and test data,
    the actual process takes place here includes:
    1. reshape to nofrows, 32, 32, 3
    2. shuffle
    3. dummify labels
    """
    x_train = sio.loadmat("train_32x32.mat")['X'] #change the path when needed
    y_train = sio.loadmat("train_32x32.mat")['y']
    x_test = sio.loadmat("test_32x32.mat")['X']
    y_test = sio.loadmat("test_32x32.mat")['y']
    
    print(x_train.shape)

    # input image dimensions
    img_rows, img_cols = 32, 32
    num_classes = 10 # starts with 1 not 0
    #the data, shuffled and split between train and test sets # distorted!!!
    #x_train = x_train.reshape(x_train.shape[3], img_rows, img_cols, 3)
    #x_test = x_test.reshape(x_test.shape[3], img_rows, img_cols, 3)

    X_train = []
    for i in xrange(x_train.shape[3]):
        X_train.append(x_train[:,:,:,i])
    X_train = np.asarray(X_train)
    
    X_test = []
    for i in xrange(x_test.shape[3]):
        X_test.append(x_test[:,:,:,i])
    X_test = np.asarray(X_test)

    y_test1 = y_test.reshape((26032, ))
    y_test1 = [y-1 for y in y_test1]
    y_train1 = y_train.reshape((73257, ))
    y_train1 = [y-1 for y in y_train1]    
    
    input_shape = (img_rows, img_cols, 3)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_test /= 255
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train2 = keras.utils.to_categorical(y_train1, num_classes)
    y_test2 = keras.utils.to_categorical(y_test1, num_classes)
    
    y_train2 = y_train2.astype('int32')
    y_test2 = y_test2.astype('int32')

    
    print("after process: X train shape: {}, X test shape: {}, y train shape: {}, y test shape: {}".format(x_train.shape, x_test.shape, y_train2.shape, y_test2.shape))
    return  input_shape, X_train, X_test, y_train2, y_test2


input_shape, x_train, x_test, y_train, y_test = svhn_data_split()

# baseline
def baseline_train_cnn_schn(input_shape, x_train, y_train, x_test, y_test):
    num_classes=10
    nb_classes = 10
    batch_size = 128
    epochs = 30
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                      activation='relu',
                      input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization(epsilon=0.001,  axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_regularizer=None, beta_regularizer=None))
    model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(BatchNormalization(epsilon=0.001,  axis=-1, momentum=0.99, weights=None, beta_init='zero',  gamma_regularizer=None, beta_regularizer=None))
    model.add(Dense(128, activation='relu'))
    #model.add(BatchNormalization(epsilon=0.001,  axis=-1, momentum=0.99, weights=None, beta_init='zero',  gamma_regularizer=None, beta_regularizer=None))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))


    score = model.evaluate(x_test, y_test, verbose=0)
    print('Baseline Test loss:', score[0])
    print('Bseline Test accuracy:', score[1])
    
baseline_train_cnn_schn(input_shape, x_train, y_train, x_test, y_test)


def low_gamma_init(shape, name=None):
    #value = np.random.normal(loc=0.0, scale=0.05, size=shape)
    value = 0.2 * np.ones(shape)
    return K.variable(value, name=name)

def train_cnn_schn(input_shape, x_train, y_train, x_test, y_test):
    num_classes=10
    nb_classes = 10
    batch_size = 128
    epochs = 40
    model = Sequential() # oscar's

    model.add(Conv2D(32, 3, 3, border_mode='same',
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero',  gamma_regularizer=None, beta_regularizer=None))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero',  gamma_regularizer=None, beta_regularizer=None))
    model.add(Activation('relu'))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])


    datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=epochs,
                        validation_data=(x_test, y_test))


    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
train_cnn_schn(input_shape, x_train, y_train, x_test, y_test)
