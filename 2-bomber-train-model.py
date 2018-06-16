
# from grabscreen import grab_screen
# import cv2
# import time
# import os
# import pandas as pd
# from tqdm import tqdm
# from collections import deque
# # from models import inception_v3 as googlenet
# # from models import inception_v3 as googlenet

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from random import shuffle

from models import inception_v3 as googlenet

FILE_I_END = 1860

# WIDTH = 480
# HEIGHT = 270
WIDTH = int( 640 / 2 )
HEIGHT = int( 480 / 2 )
LR = 1e-3
EPOCHS = 19

MODEL_NAME = ''
PREV_MODEL = ''

LOAD_MODEL = True

# wl = 0
# sl = 0
# al = 0
# dl = 0
#
# wal = 0
# wdl = 0
# sal = 0
# sdl = 0
# nkl = 0
#
# w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
# s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
# a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
# d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
# wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
# wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
# sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
# sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
# nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]



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



print("here1")

with tf.Graph().as_default():

    # module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/classification/1")
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/classification/1")



    print("here2")

    height, width = hub.get_expected_image_size(module)


    print("here3")

    # model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)
    # model = module(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)
    model = module(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)

    if LOAD_MODEL:
        model.load(PREV_MODEL)
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

                ##            # [   [    [FRAMES], CHOICE   ]    ]
                ##            train_data = []
                ##            current_frames = deque(maxlen=HM_FRAMES)
                ##
                ##            for ds in data:
                ##                screen, choice = ds
                ##                gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
                ##
                ##
                ##                current_frames.append(gray_screen)
                ##                if len(current_frames) == HM_FRAMES:
                ##                    train_data.append([list(current_frames),choice])

                # #
                # always validating unique data:
                # shuffle(train_data)
                train = train_data[:-50]
                test = train_data[-50:]

                X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
                Y = [i[1] for i in train]

                test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
                test_y = [i[1] for i in test]

                # model.fit({'input': X}, {'targets': Y}, n_epoch=1,
                #           validation_set=({'input': test_x}, {'targets': test_y}),
                #           snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
                model.fit({'input': X}, {'targets': Y}, n_epoch=1,
                          validation_set=({'input': test_x}, {'targets': test_y}),
                          snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())

                    print(sess.run(model))

                if count % 10 == 0:
                    print('SAVING MODEL!')
                    model.save(MODEL_NAME)

            except Exception as e:
                print(str(e))

    #

    # tensorboard --logdir=foo:J:/phase10-code/log

