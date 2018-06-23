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
from keras import optimizers
from keras import losses
import keras

import sys

from IPython.utils.capture import capture_output
import keyboard

from keras.models import load_model
from keras.models import Model

from grabscreen import grab_screen

# FILE_I_END = 19
FILE_I_END = 5

# WIDTH = 480
# HEIGHT = 270
WIDTH = int( 640 / 2 )
HEIGHT = int( 480 / 2 )
# 320 * 240

LR = 1e-3
# LR = 1e-4
# EPOCHS = 30
# EPOCHS = 1
EPOCHS = 10

MODEL_NAME = 'bomberman-nn-keras_v15_5classes.h5'
PREV_MODEL = MODEL_NAME

LOAD_MODEL = True

# outer NN variables ?
e1 =     0
d1 =     0
s1 =     0
f1 =     0
ctrl1 =  0
shift1 = 0

# default player keyboard binding
e =     [1,0,0,0,0]
d =     [0,1,0,0,0]
s =     [0,0,1,0,0]
f =     [0,0,0,1,0]
ctrl =  [0,0,0,0,1]

from keras.layers.core import Activation


# your desktop resolution:
desktopHeight = 1080
desktopWidth = 1920

# 640*480
HEIGTH = 480
WIDTH = 640


# calculating top left anchor point in window mode:
anchorHeightTopLeft = int( ( desktopHeight - HEIGTH ) / 2 )
anchorHWidthTopLeft = int( ( desktopWidth - WIDTH ) / 2 )

anchorHeightBotRight = anchorHeightTopLeft + HEIGTH
anchorWidthBotRight = anchorHWidthTopLeft + WIDTH


if LOAD_MODEL:
    model = keras.models.load_model(PREV_MODEL)
    print('We have loaded a previous model!!!!')

def detect_key_from_frame():
    # model.pre
    time.sleep(1)
    pass

paused = True

while (True):
    if (paused == True):
        time.sleep(1)
        paused = False
    # DONE: find the proper anchor
    while paused == False:
        # predict(self, x, batch_size=None, verbose=0, steps=None)
        # print("capturing")
        last_time = time.time()


        # getting the window mode screen
        screen = grab_screen(region=(anchorHWidthTopLeft, anchorHeightTopLeft,
                                     anchorWidthBotRight, anchorHeightBotRight))

        # pixels characters 2
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (int(WIDTH / 2), int(HEIGTH / 2)))
        # run a color convert:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        if (keyboard.is_pressed('p') == True):
            paused = True


        # model.predict(self, x, batch_size=None, verbose=0, steps=None)
        # prediction = model.predict(screen, batch_size=None, verbose=0, steps=None)

        # print("screen.shape",screen.shape)
        # print()

        # print("screen[:,:,0].shape:",screen[:,:,0].shape)
        screen_reshaped = (screen[:,:,0]).reshape((1,76800))

        prediction = model.predict(screen_reshaped, batch_size=1, verbose=0, steps=None)

        print("prediction:",prediction)

        # time.sleep(1)

        # print("loop time :{}ms".format((time.time()-last_time)*1000))



