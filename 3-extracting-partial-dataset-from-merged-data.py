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
FILE_I_END = 1

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

# outputsOccurence = np.zeros((numberOfDifferentsOutput,1))

while(iii<FILE_I_END+1):
    print("====================================")
    # file_name = './phase7-larger-color/training_data-{}.npy'.format(iii)

    file_name = './phase7-larger-color-merged/training_data_merged-{}.npy'.format(iii)

    # full file info
    # train_data=[]

    # with load('foo.npz') as data:
    #     a = data['a']
    # with np.load(file_name) as data:
    #     train_data = data['a']

    train_data = np.load(file_name)

    # fd = os.open(file_name, os.O_RDWR | os.O_CREAT)
    # fo = os.fdopen(fd, "r")
    # fo.close()

    # train_data = np.load(file_name, mmap_mode='r')
    # train_data = sc.misc.imread(file_name)
    print('training_data-{}.npy'.format(iii), len(train_data))

    lenghtInt = len((train_data[:,1])[0])

    possibleOutputs = np.zeros((lenghtInt, lenghtInt))

    i5 = 0
    while (i5 < lenghtInt):
        possibleOutputs[i5, i5] = 1;
        i5 += 1

    print("possibleOutputs:\n", possibleOutputs)

    # outputsOccurence = np.zeros((numberOfDifferentsOutput, 1))

    outputsOccurence = np.empty(lenghtInt, dtype=np.object)

    # trying to do automatic generation
    # outputsOccurence = [[],[],[],[],[],[]]
    # outputsOccurence = [[] for i in range(6)]
    outputsOccurence = [[] for i in range(lenghtInt)]

    outputs_number = 0
    for Outputs in possibleOutputs:
        print("Outputs:",Outputs)
        i6 = 0
        # test_argwhere = np.argwhere((train_data[:, 1])[i6]!=0)
        # print()

        while(i6<len(train_data)):
            tmpArray= np.asarray((train_data[:, 1])[i6])
            tmpArray2= np.asarray(Outputs)
            # print("tmpArray:",tmpArray)
            # print("tmpArray2:",tmpArray2)
            if(np.array_equal(tmpArray,tmpArray2)):
                # print("np.array_equal(tmpArray,tmpArray2)")
                # print("i6:",i6)
                # print("Outputs:",Outputs)
                outputsOccurence[outputs_number].append([i6])
            i6 += 1
        print("len(outputsOccurence["+str(outputs_number)+"]):",
               len(outputsOccurence[outputs_number]))
        outputs_number += 1


    print()

        # resOutputs = train_data[:,1]
        # resOutputs = train_data[np.where(train_data==Outputs)]
        # print("resOutputs:",resOutputs)

    #
    # print("train_data.shape:",train_data.shape)
    # print("train_data[:,0].shape:",train_data[:,0].shape)
    # print("train_data[:,1].shape:",train_data[:,1].shape)

    # print(train_data[:, 0].reshape((500,)).shape)
    # tmpScreen +=train_data[:, 0]
    # tmpOutput +=train_data[:, 1]
    # test= np.append(tmpData, train_data)
    # print(type(tmpData))
    # tmpData = np.append(tmpData, train_data)

    # tmpScreen= np.append(tmpScreen,train_data[:, 0])
    # tmpOutput= np.append(tmpOutput,train_data[:, 1])
    # print("tmpScreen.shape:",tmpScreen.shape)
    # print("tmpOutput.shape:",tmpOutput.shape)


    # if iii % 10 == 0:
    #     training_data = []
    #     training_data.append([np.asarray(tmpScreen), np.asarray(tmpOutput)])
    #
    #     print("len(training_data):",len(training_data))
    #     # print("(np.asarray(training_data_numpy_array)[:,:,:,:,0]).shape:",
    #     #        (np.asarray(training_data_numpy_array)[:,:,:,:,0]).shape)
    #     #
    #     # print("training_data_numpy_array[:,:,:,:,0].shape:",
    #     #        training_data_numpy_array[:,:,:,:,0].shape)
    #     #
    #     # training_data is a list
    #     print("len(training_data[0]):",
    #            len(training_data[0]))
    #     print("(np.asarray(training_data[0])).shape",
    #            (np.asarray(training_data[0])).shape)
    #     print("(np.asarray(training_data[0])).reshape((5000,2)).shape",
    #            (np.asarray(training_data[0])).reshape((5000,2)).shape)
    #
    #     testLOL = (np.asarray(training_data[0])).transpose()
    #
    #     print("len(training_data):",len(training_data))
    #     print("len(training_data[:,0]):",len(training_data[:]))
    #     # training_data.append(train_data)
    #     file_name_merged = './phase7-larger-color-merged/training_data_merged-{}.npy'.format(int(iii/10))
    #     # np.save(file_name_merged, training_data)
    #     np.save(file_name_merged, testLOL)
    #     print("len(tmpData):",len(tmpData))
    #     tmpData = []
    #     tmpScreen = []
    #     tmpOutput = []



    # time.sleep(1)
    iii += 1

