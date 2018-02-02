import sys
import os
from os import path
import pandas as pd
import numpy as np
sys.path.append(path.abspath('./util'))
from clearMail import Mail2txt

np.random.seed(7)

print("\n....Reading Data Set....\n")
directory="./CSDMC2010_SPAM/CSDMC2010_SPAM/TRAINING/"
holdtext = Mail2txt(directory)

#....Extrating Features.....................
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=3000)
vect.fit(holdtext)
simple_train_dtm = vect.transform(holdtext)
std   =  simple_train_dtm.toarray()
X =std

#....load labels..........................
path="./CSDMC2010_SPAM/CSDMC2010_SPAM/SPAMTrain.label"
td = pd.read_csv(path)
label=[]
with open(path) as m:
    te = m.readlines()
    
for i in range(len(te)):
    lb = te[i].split(" ")
    label.append(int(lb[0]))

y = np.zeros(shape=(len(label),1))
i=0
for e in label:
    y[i] = e
    i+=1

#....Train & Test DataSet...................
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)

print("\n\nTrain Set   = ",X_train.shape)
print("Train label = ",y_train.shape)
print("Test Set    = ",X_test.shape)
print("Test label  = ",y_test.shape)

#....Create NN model......................
print('\n....Creating NN Model....\n')
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(units=1500, input_dim=3000))
model.add(Activation('relu'))
model.add(Dense(units=750))
model.add(Activation('relu'))
model.add(Dense(units=180))
model.add(Activation('relu'))
model.add(Dense(units=42))
model.add(Activation('relu'))
model.add(Dense(units=9))
model.add(Activation('relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

print("\n....Training NN Model....\n")
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=64)
print('\n....Testing NN model....\n')
result1 = model.evaluate(X_test, Y_test, batch_size=len(y_test))
model.save('my_model.h5')
print("\nResult [loss,metrics] = ",result1)


