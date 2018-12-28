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

# MODEL_NAME = 'bomberman-nn-keras_v14_5classes.h5'
# MODEL_NAME = 'bomberman-nn-keras_v15_5classes.h5'
# MODEL_NAME = 'bomberman-nn-keras_v16_5classes.h5'
# MODEL_NAME = 'bomberman-nn-keras_v13_5classes_data_p_255.h5'
MODEL_NAME = 'bomberman-nn-keras_v26_6classes.h5'
# MODEL_NAME = 'BasicCNN-5-epochs-0.0001-LR-STAGE1-0-.h5'
# MODEL_NAME = 'BasicCNN-5-epochs-0.0001-LR-STAGE1-4-.h5'

# MODEL_NAME = 'bomberman-nn-keras_v17_5classes.h5'

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

all_keys=[e,d,s,f,ctrl]

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

previousPrediction = [[]]

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
        # screen_reshaped = (screen[:,:,0]).reshape((1,76800))
        # prediction = model.predict(screen_reshaped/255, batch_size=1, verbose=0, steps=None)

        screen_reshaped = screen.reshape((-1,240,320,3))
        prediction = model.predict(screen_reshaped, batch_size=1, verbose=0, steps=None)

        # if(np.array_equal(prediction,previousPrediction)==False):
        #     print("prediction:", prediction)
        # else:
        #     print("lol")
        #
        # print("prediction:", prediction)
        print(prediction)


        previousPrediction = prediction

        # print("prediction:",prediction)

        # print("loop time :{}ms".format((time.time()-last_time)*1000))

        # print(type(screen))
        # too see what is captured
        cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('t'):
            cv2.destroyAllWindows()
            break

        if(1):
            mostImportantInputNumber = (np.where(prediction[0]==np.max(prediction[0])))[0]

            # futureKeypress = np.zeros((1,5), np.uint8)
            futureKeypress = np.zeros((1,6), np.uint8)
            futureKeypress[0][mostImportantInputNumber] = 1

            tmpInt2 = 0

            once = False

            for futureInput in all_keys:
                if(once==False):
                    once=True
                    if(mostImportantInputNumber==0):
                        # print("e")
                        keyboard.press('e')
                        time.sleep(0.15)
                        # time.sleep(0.05)
                        keyboard.release('e')
                        # pass
                    if(mostImportantInputNumber==1):
                        print("d")
                        keyboard.press('d')
                        time.sleep(0.15)
                        # time.sleep(0.05)
                        keyboard.release('d')
                        # pass
                    if(mostImportantInputNumber==2):
                        print("s")
                        keyboard.press('s')
                        time.sleep(0.15)
                        # time.sleep(0.05)
                        keyboard.release('s')
                        # pass
                        pass
                    if(mostImportantInputNumber==3):
                        print("f")
                        keyboard.press('f')
                        time.sleep(0.15)
                        # time.sleep(0.05)
                        keyboard.release('f')
                        # pass
                    if(mostImportantInputNumber==4):
                        print("ctrl")
                        keyboard.press('ctrl')
                        time.sleep(0.15)
                        # time.sleep(0.05)
                        keyboard.release('ctrl')
                        # pass
                    if(mostImportantInputNumber==5):
                        time.sleep(0.15)



                tmpInt2 +=1

            # time.sleep(0.05)
