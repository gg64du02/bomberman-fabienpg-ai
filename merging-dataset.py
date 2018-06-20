import numpy as np

import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
# from models import inception_v3 as googlenet
from random import shuffle

import sys

# FILE_I_END = 19
FILE_I_END = 11

# WIDTH = 480
# HEIGHT = 270
WIDTH = int( 640 / 2 )
HEIGHT = int( 480 / 2 )
# 320 * 240

LR = 1e-3
# LR = 1e-4
# EPOCHS = 30
EPOCHS = 500

iii = 1

tmpData = []
tmpScreen = []
tmpOutput = []
# tmpData = np.zeros((500,1))
# tmpScreen = np.zeros((500,1))
# tmpOutput = np.zeros((500,1))

while(iii<FILE_I_END+1):
    print("====================================")
    file_name = './phase7-larger-color/training_data-{}.npy'.format(iii)
    # full file info
    # train_data=[]

    # with load('foo.npz') as data:
    #     a = data['a']
    # with np.load(file_name) as data:
    #     train_data = data['a']

    train_data = np.load(file_name)

    fd = os.open(file_name, os.O_RDWR | os.O_CREAT)
    fo = os.fdopen(fd, "r")
    fo.close()

    # train_data = np.load(file_name, mmap_mode='r')
    # train_data = sc.misc.imread(file_name)
    print('training_data-{}.npy'.format(iii), len(train_data))

    print("train_data.shape:",train_data.shape)
    print("train_data[:,0].shape:",train_data[:,0].shape)
    print("train_data[:,1].shape:",train_data[:,1].shape)

    # print(train_data[:, 0].reshape((500,)).shape)
    # tmpScreen +=train_data[:, 0]
    # tmpOutput +=train_data[:, 1]
    # test= np.append(tmpData, train_data)
    # print(type(tmpData))
    # tmpData = np.append(tmpData, train_data)

    tmpScreen= np.append(tmpScreen,train_data[:, 0])
    tmpOutput= np.append(tmpOutput,train_data[:, 1])
    print("tmpScreen.shape:",tmpScreen.shape)
    print("tmpOutput.shape:",tmpOutput.shape)


    if iii % 10 == 0:
        training_data = []
        training_data.append([np.asarray(tmpScreen), np.asarray(tmpOutput)])
        print("len(training_data):",len(training_data))
        print("len(training_data[:,0]):",len(training_data[:]))
        # training_data.append(train_data)
        file_name_merged = './phase7-larger-color-merged/training_data_merged-{}.npy'.format(int(iii/10))
        # np.save(file_name_merged, training_data)
        print("len(tmpData):",len(tmpData))
        tmpData = []
        tmpScreen = []
        tmpOutput = []



    # time.sleep(1)
    iii += 1

