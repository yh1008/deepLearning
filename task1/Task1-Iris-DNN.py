# Task 1 Run a multilayer perceptron (feed forward neural network) with two hidden layers and rectifies linear nonlineartiries on the iris dataset using the Keras Sequential interface. include code for model selection and evalutation on an independent test-set.
# [4pts for running model, 3pts for correct architecture, 3pts for evaluation]

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
#import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
#import Image               
from sklearn import datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target

print (X.shape, y.shape)

def mytrain_test_split(X,y):
    r"""perform train test split
    Returns
    -------
    X_train : array
    y_train : array
    X_test : array
        dummified 
    y_test : array
        dummified
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    y_train_dummy = keras.utils.to_categorical(y_train)
    y_test_dummy = keras.utils.to_categorical(y_test)
    return X_train, X_test, y_train_dummy, y_test_dummy


def train_nnet(X_train, X_test, y_train, y_test):
    r"""
    Returns
    -------
    hisotry_callback : <class 'keras.callbacks.History'>
        used for retrospective examiniation
    """
    
    num_class = 3
    input_shape = (4,)
    epochs = 12
    
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])
    history_callback = model.fit(X_train, 
              y_train, 
              verbose=0, 
              batch_size=1,
              validation_split=0.1,
              epochs=epochs
              )
    
    score = model.evaluate(X_test, y_test, verbose=0)
    
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))
    #print(model.summary())

    return history_callback


def plot_history(logger):

    df = pd.DataFrame(logger)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[["loss", "val_loss"]].plot(linestyle="--", ax=plt.twinx())
    plt.ylabel("loss")


X_train, X_test, y_train, y_test = mytrain_test_split(X, y)

callback = train_nnet(X_train, X_test, y_train, y_test)


# get_ipython().magic(u'matplotlib inline')
plot_history(callback.history)
plt.savefig('Irish_learning_curve.png')
#Image.open('Irish_learning_curve.png').save('Irish_learning_curve.jpg','JPEG')
# #### On this graph, the accuracy keeps increasing, so does validation score. 

# Model Selection
def make_nnet(hidden_size=16):
    r"""
    Returns
    -------
    model : model itself
    """
    
    num_class = 3
    input_shape = (4,)
    
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=input_shape, activation="relu"))
    model.add(Dense(int(hidden_size/2), activation="relu"))
    model.add(Dense(num_class, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=['accuracy'])

    return  model


clf = KerasClassifier(make_nnet)
param_grid = {'epochs': [8, 20, 50, 100], 'hidden_size': [12], 'batch_size':[10]}
grid = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print ("best params: {}".format(grid.best_params_))
print ("mean test score: {} ".format(grid.cv_results_['mean_test_score']))
print ("mean train score: {} ".format(grid.cv_results_['mean_train_score']))
# res = pd.DataFrame(grid.cv_results_)
# res.pivot_table(index=["param_epochs"],
#                values=['mean_train_score', "mean_test_score"])


# #### Conclusion:
# By fixing the number of units in the first layer to 12, and the second layer units as 6, and changing number of epochs, we see 100 epochs yields the highest accuracy, which is around 0.97
