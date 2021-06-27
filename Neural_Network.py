import numpy as np
from numpy import unique
from numpy import argmax
import math
from pandas import read_csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from matplotlib import pyplot
from sklearn import preprocessing

def modelRun(data, e, bs, v):
    # np.savetxt("C:\\Users\\16479\\Desktop\\foo.csv", dataset, delimiter=",")
    # split into input (X) and output (y) variables
    # dataframe = read_csv("C:\\Users\\16479\\Desktop\\Matz_Audible.csv", header=0)
    # dataset = dataframe.values
    data_X, data_y = data[:, 1:-1], data[:, -1]
    data_X, data_y = data_X.astype('float'), data_y.astype('float')
    n_features = data_X.shape[1]
    data_X = preprocessing.scale(data_X)
    # X_train, X_test, y_train, y_test = data_X[113:], data_X[0:113], data_y[113:], data_y[0:113]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=2)

    data_y = LabelEncoder().fit_transform(data_y)
    n_class = len(unique(data_y))

    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)

    model = Sequential()
    model.add(Dense(90, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(70, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam")
    # fit the keras model on the dataset
    # Optimizers: adam, SGD, RMSprop, adagard
    # loss: sparse categorical crossentropy,
    # different batch sizes

    history = model.fit(X_train, y_train, epochs=e, batch_size=bs, verbose=v)

    # evaluate on test set
    yhat = model.predict(X_test)
    probs = model.predict_proba(X_test)

    yhat = argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    return acc, yhat, y_test, probs, model
    # print('Accuracy: %.3f' % acc)

def modelRunGlobal(data, e, bs, v, op):
    # np.savetxt("C:\\Users\\16479\\Desktop\\foo.csv", dataset, delimiter=",")
    # split into input (X) and output (y) variables
    # dataframe = read_csv("C:\\Users\\16479\\Desktop\\Matz_Audible.csv", header=0)
    # dataset = dataframe.values
    data_X, data_y = data[:, 1:-1], data[:, -1]
    data_X, data_y = data_X.astype('float'), data_y.astype('float')
    n_features = data_X.shape[1]
    data_X = preprocessing.scale(data_X)
    X_train, X_test, y_train, y_test = data_X[10000:], data_X[0:10000], data_y[10000:], data_y[0:10000]

    data_y = LabelEncoder().fit_transform(data_y)
    n_class = len(unique(data_y))

    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)

    model = Sequential()
    model.add(Dense(90, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(70, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=op)
    # fit the keras model on the dataset
    # Optimizers: adam, SGD, RMSprop, adagard
    # loss: sparse categorical crossentropy,
    # different batch sizes

    history = model.fit(X_train, y_train, epochs=e, batch_size=bs, verbose=v)

    # evaluate on test set
    yhat = model.predict(X_test)
    probs = model.predict_proba(X_test)

    yhat = argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    return acc, yhat, y_test, probs, model
    # print('Accuracy: %.3f' % acc)

def modelRunTest(data, test, e, bs, v, opt):
    data_X, data_y = data[:, 1:-1], data[:, -1]
    data_X, data_y = data_X.astype('float'), data_y.astype('float')

    data_XT, data_yT = test[:, 1:-1], test[:, -1]
    data_XT, data_yT = data_XT.astype('float'), data_yT.astype('float')

    n_features = data_X.shape[1]
    data_X = preprocessing.scale(data_X)
    data_XT = preprocessing.scale(data_XT)
    X_train = data_X
    X_test = data_XT,
    y_train = data_y
    y_test =  data_yT

    data_y = LabelEncoder().fit_transform(data_y)
    n_class = len(unique(data_y))

    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)

    model = Sequential()
    model.add(Dense(90, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(70, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    # fit the keras model on the dataset
    # Optimizers: adam, SGD, RMSprop, adagard
    # loss: sparse categorical crossentropy,
    # different batch sizes

    model.fit(X_train, y_train, epochs=e, batch_size=bs, verbose=v)

    # evaluate on test set
    yhat = model.predict(X_test)
    probs = model.predict_proba(X_test)

    yhat = argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    return acc, yhat, y_test, probs, model

def modelRunTrain(data, e, bs, v):
    data_X, data_y = data[:, 1:-1], data[:, -1]
    data_X, data_y = data_X.astype('float'), data_y.astype('float')


    n_features = data_X.shape[1]
    data_X = preprocessing.scale(data_X)
    X_train, y_train = data_X, data_y

    data_y = LabelEncoder().fit_transform(data_y)
    n_class = len(unique(data_y))

    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)

    model = Sequential()
    model.add(Dense(90, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(70, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer="RMSprop")
    # fit the keras model on the dataset
    # Optimizers: adam, SGD, RMSprop, adagard
    # loss: sparse categorical crossentropy,
    # different batch sizes

    history = model.fit(X_train, y_train, epochs=e, batch_size=bs, verbose=v)

    return model


def modelRunTestSingle(data, test, e, bs, v):
    x, y = np.shape(test)
    data_X, data_y = data[:, 1:-1], data[:, -1]
    data_X, data_y = data_X.astype('float'), data_y.astype('float')

    data_XT, data_yT = test[:, 1:y - 1], test[:, y - 1]
    data_XT, data_yT = data_XT.astype('float'), data_yT.astype('float')

    n_features = data_X.shape[1]
    data_X = preprocessing.scale(data_X)
    data_XT = preprocessing.scale(data_XT)
    X_train, X_test, y_train, y_test = data_X, data_XT, data_y, data_yT

    data_y = LabelEncoder().fit_transform(data_y)
    n_class = len(unique(data_y))

    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)

    model = Sequential()
    model.add(Dense(90, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(70, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer="RMSprop")
    # fit the keras model on the dataset
    # Optimizers: adam, SGD, RMSprop, adagard
    # loss: sparse categorical crossentropy,
    # different batch sizes

    history = model.fit(X_train, y_train, epochs=e, batch_size=bs, verbose=v)

    # evaluate on test set
    yhat = model.predict(X_test)
    probs = model.predict_proba(X_test)

    yhat = argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    return acc, yhat, y_test, probs, model


