import numpy as np

# forcing tf-cpu put this before tf and keras import
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import time
import os
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

# convnet

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard


# https://keras.io/losses/
# https://keras.io/optimizers/
# https://keras.io/models/model/

from keras.layers import Cropping2D

# FILE_I_END = 19
FILE_I_END = 300

# WIDTH = 480
# HEIGHT = 270
WIDTH = int( 640 / 2 )
HEIGHT = int( 480 / 2 )
# 320 * 240

# LR = 1e-3
LR = 1e-4
# EPOCHS = 1
EPOCHS = 5

MODEL_NAME = 'bomberman-nn-keras_v27_6classes.h5'
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

import random
random.seed(1)

model = Sequential()

model.add(Cropping2D(cropping=8,
                     input_shape=(240, 320, 3)))
# model.add(Conv2D(8, (8, 8), activation='softmax'))
model.add(Conv2D(16, (16, 16), activation='softmax'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (16, 16), activation='softmax'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (16, 16), activation='softmax'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.1666))

model.add(Flatten())
# model.add(Dense(512, activation='softmax'))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='softmax'))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


if LOAD_MODEL:
    del model
    model = keras.models.load_model(PREV_MODEL)
    print('We have loaded a previous model!!!!')

orig_stdout = sys.stdout
f = open('last_log.txt', 'w')
sys.stdout = f

TRYTORESUMETRAINING = False

epoch_count = -1
file_count = -1

if(TRYTORESUMETRAINING  == True):
    print('if(TRYTORESUMETRAINING  == True):')
    # !/usr/bin/env python
    from stat import S_ISREG, ST_CTIME, ST_MODE
    import os, sys, time
    # path to the directory (relative or absolute)
    # dirpath = sys.argv[1] if len(sys.argv) == 2 else r'.'
    dirpath = r'.'

    # get all entries in the directory w/ stats
    entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
    entries = ((os.stat(path), path) for path in entries)

    # leave only regular files, insert creation date
    entries = ((stat[ST_CTIME], path)
               for stat, path in entries if S_ISREG(stat[ST_MODE]))
    #NOTE: on Windows `ST_CTIME` is a creation date
    #  but on Unix it could be something else
    #NOTE: use `ST_MTIME` to sort by a modification date

    for cdate, path in sorted(entries):
        print(time.ctime(cdate), os.path.basename(path))
        tmpStr = os.path.basename(path)
        # print(tmpStr)
        # print(tmpStr[0:8] == 'BasicCNN')
        if(tmpStr[0:8] == 'BasicCNN'):
            print(tmpStr)
            tmpStr2 = tmpStr.split('.')[1]
            epoch_count = tmpStr2.split('-')[3]
            file_count = tmpStr2.split('-')[4]

epoch_count = int(epoch_count)
file_count = int(file_count)

print('epoch_count',epoch_count)
print('file_count',file_count)

def generate_arrays_from_folder(folder):
# def generate_arrays_from_folder(path):
    folderNameStr = 'dataset'
    for e in range(EPOCHS):

        global epoch_count
        if(epoch_count!=-1):
            print('e',e)
            if(e!=epoch_count):
                continue
                # epoch_count = -1
            else:
                epoch_count = -1

        data_order = [i for i in range(1, FILE_I_END + 1)]
        print("EPOCHS:",e)
        for count, i in enumerate(data_order):

            global file_count
            if (file_count != -1):
                print('i', i)
                if (i != file_count):
                    continue
                    # file_count = -1
                else:
                    file_count = -1

            try:
                file_name = './'+folder+'/training_data_merged-partial-dataset-{}.npy'.format(i)
                file_name = './'+folder+'/training_data_merged-{}.npy'.format(i)
                print("file_name:",file_name)

                # full file info
                train_data = np.load(file_name)

                # # testing if putting a memory stick might help
                # train_data = train_data[0:int(len(train_data)/2)]
                # train_data = train_data[0:int(len(train_data)/2)]
                # lol = int( ( len(train_data) - len(train_data) % 100 ) / 100 )
                # train_data = train_data[0: lol - 1 ]
                # train_data = train_data[0: 2000 ]

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
                one_hot_labels = keras.utils.to_categorical(labels, num_classes=6)
                print("one_hot_labels:", one_hot_labels.shape)

                print("data.shape:",data.shape)

                # yield(data/255, one_hot_labels)

                print("data.shape:",data.shape)

                print("len(one_hot_labels):",len(one_hot_labels))
                # len(one_hot_labels): 3225
                print("len(one_hot_labels[0]):",len(one_hot_labels[0]))
                # len(one_hot_labels[0]): 5

                # test_size = len(train_data)
                test_size = 100

                x_train = np.array([i[0] for i in train_data[:-test_size]]).reshape(-1, 240, 320, 3)
                y_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1,6)


                x_test = np.array([i[0] for i in train_data[-test_size:]]).reshape(-1, 240, 320, 3)
                y_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1,6)

                # # normalization: make it easier for the CNN to learn ?
                # x_train = tf.keras.utils.normalize(x_train, axis=1)
                # x_test = tf.keras.utils.normalize(x_test, axis=1)

                model.fit(x_train, y_train,
                          batch_size=36,
                          validation_data=(x_test, y_test),
                          shuffle=True,
                          verbose=1)

                if(count%10==0):
                    model.save("BasicCNN-{}-epochs-{}-LR-STAGE1-{}-{}.h5".format(EPOCHS, learning_rate, e, count))
                # model.save("BasicCNN-{}-epochs-{}-LR-STAGE1-{}-{}.h5".format(EPOCHS, learning_rate, e,count))

                print("BasicCNN-{}-epochs-{}-LR-STAGE1-{}-.h5".format(EPOCHS, learning_rate, e),"saved")

                pass
            except Exception as ex:
                print(str(ex))
                print(sys.exc_info())


# model.fit_generator(generate_arrays_from_folder('phase-3'),
#                     steps_per_epoch=5, epochs=EPOCHS)


generate_arrays_from_folder('phase-2')

model.save(MODEL_NAME)

sys.stdout = orig_stdout
f.close()