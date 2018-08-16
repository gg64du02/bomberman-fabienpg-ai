import numpy as np

# forcing tf-cpu put this before tf and keras import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

# convnet

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard


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

# LR = 1e-3
LR = 1e-4
# EPOCHS = 1
EPOCHS = 5

MODEL_NAME = 'bomberman-nn-keras_v16_5classes.h5'
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


model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',
#                  input_shape=(176, 200, 3),
#                  activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(240, 320, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
# model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
# model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE1")


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

                print("train_data[0,0].shape:",train_data[0,0].shape)
                print("train_data[0,1]:",train_data[0,1])

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

                # Convert labels to categorical one-hot encoding
                one_hot_labels = keras.utils.to_categorical(labels, num_classes=5)
                print("one_hot_labels:", one_hot_labels.shape)

                print("data.shape:",data.shape)

                # yield(data/255, one_hot_labels)

                print("data.shape:",data.shape)
                # data.shape: (3225, 76800)
                print("len(one_hot_labels):",len(one_hot_labels))
                # len(one_hot_labels): 3225
                print("len(one_hot_labels[0]):",len(one_hot_labels[0]))
                # len(one_hot_labels[0]): 5

                # test_size = len(train_data)
                test_size = 100

                x_train = np.array([i[0] for i in train_data[:-test_size]]).reshape(-1, 240, 320, 3)
                y_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1,5)

                x_test = np.array([i[0] for i in train_data[-test_size:]]).reshape(-1, 240, 320, 3)
                y_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1,5)

                model.fit(x_train, y_train,
                          batch_size=100,
                          validation_data=(x_test, y_test),
                          shuffle=True,
                          verbose=1)

                model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(EPOCHS, learning_rate))

                print("lol")

                pass
            except Exception as e:
                print(str(e))
                print(sys.exc_info())


# model.fit_generator(generate_arrays_from_folder('phase-3'),
#                     steps_per_epoch=5, epochs=EPOCHS)
#
# model.save(MODEL_NAME)

generate_arrays_from_folder('phase-3')
