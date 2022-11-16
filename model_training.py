import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import xgboost as xgb
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation

def fsplit():
    tdata = pd.read_csv("C:/Users/Saisamarth/Desktop/DA/Project/tdata.csv")
    fdata = tdata.drop(columns = 'IDSong')

    X = fdata.iloc[: ,:-1].values
    y = fdata['Class'].values  

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    return x_train, x_test, y_train, y_test



def svm():
    svm_model = SVC(kernel = 'linear', C = 1).fit(x_train, y_train)
    svm_pred = svm_model.predict(x_test)
    accuracy = svm_model.score(x_test, y_test)
    print("Test accuracy of SVM Model is: {0:.2%}".format(accuracy))

def knn():
    knn = KNeighborsClassifier(n_neighbors = 4).fit(x_train, y_train)
    accuracy = knn.score(x_test, y_test)
    print("Test accuracy of KNN Model is: {0:.2%}".format(accuracy))

def guass():
    gnb = GaussianNB().fit(x_train, y_train)
    gnb_predictions = gnb.predict(x_test)
    accuracy = gnb.score(x_test, y_test)
    print("Test accuracy of Naive Bayes Model is: {0:.2%}".format(accuracy))



def fcnnsplit():
    tdata = pd.read_csv("C:/Users/Saisamarth/Desktop/DA/Project/tdata.csv")
    fdata = tdata.drop(columns = 'IDSong')

    X = fdata.iloc[: ,:-1].values
    y = fdata['Class'].values  

    encoder = OneHotEncoder()
    y = encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = fcnnsplit()

def xgboost():
    
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.fit(x_train, y_train)
    preds = xgb_cl.predict(x_test)
    print("Test accuracy of XGBoost Model is: {0:.2%}".format(accuracy_score(y_test, preds)))
    

def model_train():
    model = Sequential()
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(x_train, y_train, batch_size=32, epochs=60, validation_data=(x_test, y_test), verbose=2)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy of CNN: {0:.2%}".format(score[1]))
    model.save("model.h5")


model_train()
xgboost()

x_train, x_test, y_train, y_test = fsplit()

svm()
knn()
guass()