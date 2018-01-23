#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:23:55 2018

@author: kishankumar
"""

# Digit Recognizer

# importing the necessary library, we'll be using MNIST dataset
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.utils import np_utils
import keras.models as km

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# the image is 28X28 and for training purposes we have 60000 data to train on and 10000 data to test on
# in total we have 70000 data

# First we will form a 784 pixels columns
total_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], total_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], total_pixels).astype('float32')

# Normalizing the input between 0-1 as they are currently between 0-255
X_train = X_train / 255
X_test = X_test / 255

# one hot encoding the inputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# reshaping the X_train and X_test so that they can be inputted to the CNN model
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convolutional Neural Network
classifier = Sequential()

# adding first convolutional layer
classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(BatchNormalization())
# adding first Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5))
# adding second convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
# adding second Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

# Artificial Neural Network
classifier.add(Dense(units = 256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=10, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Fit the model
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# save the model
km.save_model(classifier,'highly_trained.h5')



