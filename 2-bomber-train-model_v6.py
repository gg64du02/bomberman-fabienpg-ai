import numpy as np

import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
# from models import inception_v3 as googlenet
from random import shuffle
# import tensorflow as tf


from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers
from keras import losses
import keras

import sys


# https://keras.io/losses/
# https://keras.io/optimizers/
# https://keras.io/models/model/


# FILE_I_END = 19
FILE_I_END = 1

# WIDTH = 480
# HEIGHT = 270
WIDTH = int( 640 / 2 )
HEIGHT = int( 480 / 2 )
# 320 * 240

LR = 1e-3
# EPOCHS = 30
EPOCHS = 1

MODEL_NAME = 'bomberman-nn-keras.h5'
PREV_MODEL = MODEL_NAME

LOAD_MODEL = False

# outer NN variables ?
e1 =     0
d1 =     0
s1 =     0
f1 =     0
ctrl1 =  0
shift1 = 0

# default player keyboard binding
e =     [1,0,0,0,0,0]
d =     [0,1,0,0,0,0]
s =     [0,0,1,0,0,0]
f =     [0,0,0,1,0,0]
ctrl =  [0,0,0,0,1,0]
shift = [0,0,0,0,0,1]

# For a single-input model with 10 classes (categorical classification):

# https://keras.io/getting-started/sequential-model-guide/
# 320 * 240 = 76800
model = Sequential()
# 32 is probably the batch size
model.add(Dense(32, activation='relu', input_dim=76800))
# 6 is probably the number of classes
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # Generate dummy data
# import numpy as np
# data = np.random.random((1000, 100))
# labels = np.random.randint(10, size=(1000, 1))
#
# # Convert labels to categorical one-hot encoding
# one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
#
# # Train the model, iterating on the data in batches of 32 samples
# model.fit(data, one_hot_labels, epochs=10, batch_size=32)

#
if LOAD_MODEL:
    keras.models.load_model(PREV_MODEL)
    print('We have loaded a previous model!!!!')

# iterates through the training files

for e in range(EPOCHS):
    # data_order = [i for i in range(1,FILE_I_END+1)]
    data_order = [i for i in range(1, FILE_I_END + 1)]
    shuffle(data_order)
    for count, i in enumerate(data_order):

        try:
            # file_name = 'J:/phase10-random-padded/training_data-{}.npy'.format(i)
            file_name = './phase7-larger-color/training_data-{}.npy'.format(i)

            # full file info
            train_data = np.load(file_name)
            print('training_data-{}.npy'.format(i), len(train_data))

            # shuffle(train_data)

            train = train_data[:-50]
            test = train_data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = [i[1] for i in train]
            print("X.shape:",X.shape)
            print("len(Y):", len(Y))

            test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)

            # former vector
            # test_y = [i[1] for i in test]
            # for keras one hot function
            test_y = [i[1] for i in test]
            test_y2 = np.zeros((50, 1))
            tmp_i = 0
            print("count1test")
            for i in test_y:
                tmp_ii = 0
                for ii in i:
                    if(ii==1):
                        test_y2[tmp_i,0] = tmp_ii
                    tmp_ii +=1
                tmp_i +=1
            print("test_y2.shape:",test_y2.shape)


            print("test_x:",test_x.shape)
            print("len(test_y):",len(test_y))


            data = test_x[:,:,:,0].reshape((50,WIDTH*HEIGHT))
            print("data.shape:",data.shape)
            # former vector
            # labels = np.asarray(test_y)
            labels = np.asarray(test_y2)
            print("len(labels):",len(labels))
            print("labels.shape:",labels.shape)


            # from the website
            # data = np.random.random((1000, 76800))
            # labels = np.random.randint(6, size=(1000, 1))
            # print("data:",data.shape)
            # print("labels:",labels.shape)

            # Convert labels to categorical one-hot encoding
            one_hot_labels = keras.utils.to_categorical(labels, num_classes=6)
            print("one_hot_labels:",one_hot_labels.shape)

            # Train the model, iterating on the data in batches of 32 samples
            model.fit(data, one_hot_labels, epochs=10, batch_size=32)

            if count % 10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                # model.save('./test.tfmodel')

        except Exception as e:
            print(str(e))
            print(sys.exc_info())


#

# tensorboard --logdir=foo:J:/phase10-code/log

