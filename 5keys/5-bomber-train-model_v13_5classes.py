import numpy as np

import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
# from models import inception_v3 as googlenet
from random import shuffle
import tensorflow as tf


from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras import optimizers
from keras import losses
import keras

import sys

from IPython.utils.capture import capture_output


# https://keras.io/losses/
# https://keras.io/optimizers/
# https://keras.io/models/model/


# FILE_I_END = 19
FILE_I_END = 10

# WIDTH = 480
# HEIGHT = 270
WIDTH = int( 640 / 2 )
HEIGHT = int( 480 / 2 )
# 320 * 240

LR = 1e-3
# LR = 1e-4
# EPOCHS = 30
# EPOCHS = 1
# EPOCHS = 10
# EPOCHS = 120
EPOCHS = 1

MODEL_NAME = 'bomberman-nn-keras_v13_5classes.h5'
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

from keras.layers.core import Activation

# # ============================================================
# https://keras.io/getting-started/sequential-model-guide/
# 320 * 240 = 76800
model = Sequential()
# Input tensor for sequences of 32 timesteps,
model.add(Dense(200, activation='relu', input_dim=76800))
# model.add(Activation('relu'));
# grid is 15*20 32 pixels
# make it so it would be 30*40
# /64
model.add(Dense(1200, activation='relu', input_dim=1200))
keras.layers.Dropout(0.5)
# model.add(Activation('relu'));
# 6 is probably the number of classes
model.add(Dense(5, activation='softmax'))
# model.add(Activation('relu'));
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(optimizer='adagrad',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# # ============================================================


# a = Input(shape=(76800,))
# b = Dense(32)(a)
# model = Model(inputs=a, outputs=b)



if LOAD_MODEL:
    keras.models.load_model(PREV_MODEL)
    print('We have loaded a previous model!!!!')

# iterates through the training files
# init callback function fro tensor board
# use: tensorboard --logdir path_to_current_dir/Graph
# tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
#         histogram_freq=0, write_graph=True, write_images=True)
# tbCallBack = keras.callbacks.TensorBoard(log_dir='D:/Graph',
#         histogram_freq=0, write_graph=True, write_images=True)
# tbCallBack = print("tbCallBack:")

# use tensorboard on those
# dense_3_loss
# acc


def generate_arrays_from_folder(folder):
# def generate_arrays_from_folder(path):
    folderNameStr = 'dataset'
    for e in range(EPOCHS):
        data_order = [i for i in range(1, FILE_I_END + 1)]
        print("EPOCHS:",e)
        for count, i in enumerate(data_order):

            try:
                # file_name = './'+folder+'/bomberman-dataset-0.npy'
                # file_name = './'+folder+'/bomberman-dataset-{}.npy'.format(i)
                file_name = './'+folder+'/training_data_merged-partial-dataset-{}.npy'.format(i)
                print("file_name:",file_name)

                # full file info
                train_data = np.load(file_name)

                # train_data = np.load(file_name,encoding='ASCII')
                print('\ntraining_data-{}.npy'.format(i), len(train_data))

                td_length = len(train_data)
                print("td_length:", td_length)

                shuffle(train_data)

                train = train_data[:-td_length]
                test = train_data[-td_length:]

                X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
                Y = [i[1] for i in train]
                print("X.shape:", X.shape)
                print("len(Y):", len(Y))

                test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)

                # former vector
                # test_y = [i[1] for i in test]
                # for keras one hot function
                test_y = [i[1] for i in test]
                test_y2 = np.zeros((td_length, 1))
                tmp_i = 0
                # print("count1test")
                for i in test_y:
                    tmp_ii = 0
                    for ii in i:
                        if (ii == 1):
                            test_y2[tmp_i, 0] = tmp_ii
                        tmp_ii += 1
                    tmp_i += 1
                print("test_y2.shape:", test_y2.shape)

                print("test_x:", test_x.shape)
                print("len(test_y):", len(test_y))

                data = test_x[:, :, :, 0].reshape((td_length, WIDTH * HEIGHT))
                print("data.shape:", data.shape)
                # former vector
                # labels = np.asarray(test_y)
                labels = np.asarray(test_y2)
                print("len(labels):", len(labels))
                print("labels.shape:", labels.shape)

                # from the website
                # data = np.random.random((1000, 76800))
                # labels = np.random.randint(6, size=(1000, 1))
                # print("data:",data.shape)
                # print("labels:",labels.shape)

                # Convert labels to categorical one-hot encoding
                one_hot_labels = keras.utils.to_categorical(labels, num_classes=5)
                print("one_hot_labels:", one_hot_labels.shape)

                print("data.shape:",data.shape)

                # Train the model, iterating on the data in batches of 32 samples
                # model.fit(data, one_hot_labels, epochs=10, batch_size=32)

                # model.fit(data, one_hot_labels, epochs=1, verbose=0,steps_per_epoch=55)

                # yield(data,one_hot_labels)
                yield(data/255, one_hot_labels)

                pass
            except Exception as e:
                print(str(e))
                print(sys.exc_info())



# model.fit_generator(generate_arrays_from_folder('phase-3'),
#                     steps_per_epoch=EPOCHS*FILE_I_END, epochs=1)

model.fit_generator(generate_arrays_from_folder('phase-3'),
                    steps_per_epoch=FILE_I_END, epochs=EPOCHS)

model.save(MODEL_NAME)

