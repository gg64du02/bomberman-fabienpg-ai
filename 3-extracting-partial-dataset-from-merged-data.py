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
FILE_I_END = 5

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

    file_name = './phase-2/training_data_merged-{}.npy'.format(iii)

    # to get read of the error during debugging after to many file opening
    fd = os.open(file_name, os.O_RDWR | os.O_CREAT)
    fo = os.fdopen(fd, "r")
    fo.close()

    train_data = np.load(file_name)

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

    NumberOfOccurenceOutput = []

    # stir everything to get a better results when extracting on the first go.
    # shuffle(train_data)

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

        NumberOfOccurenceOutput.append(len(outputsOccurence[outputs_number]))

        outputs_number += 1

    # bare minimum to extract
    minimumToExtract = min(NumberOfOccurenceOutput)
    # preparing the numpy for file writing
    # extractedNumpyFromTrainData = np.zeros((lenghtInt*minimumToExtract,2))
    extractedNumpyFromTrainData = [[[],[]] for line in range(int(lenghtInt*minimumToExtract))]
    # extractedNumpyFromTrainData = np.zeros((int(lenghtInt*minimumToExtract),2))

    print("len(extractedNumpyFromTrainData):",len(extractedNumpyFromTrainData))

    # counting remaining
    outputsOccurenceInOutputFile = NumberOfOccurenceOutput


    # outputsOccurence[outputs_number])
    print()
    i7 = 0

    i9 = 0
    print("len(outputsOccurence):",len(outputsOccurence))
    while(i7<len(outputsOccurence)):
        # print(outputsOccurence[i7])
        print("i7:",i7)
        i8 = 0
        # lol (outputsOccurence[5])[0]
        while(i8<minimumToExtract):
            print("i8:",i8)
            # print((outputsOccurence[i7])[i8])
            # print( ( (outputsOccurence[i7])[i8] )[0]      )
            tmpIndex = ( (outputsOccurence[i7])[i8] )[0]
            print("tmpIndex:",tmpIndex)

            extractedNumpyFromTrainData[i9] = train_data[tmpIndex]

            print("i9:",i9)

            # index for tain_data
            i9 +=1

            # iteration on each occurence
            i8 +=1

        # iteration on another output
        i7+=1

    print()

    file_name_partial_extraction_dataset = './phase-3/training_data_merged-partial-dataset-{}.npy'.format(iii)

    # np.save(file_name_partial_extraction_dataset, extractedNumpyFromTrainData)

    # extractedNumpyFromTrainData is a list
    np.save(file_name_partial_extraction_dataset, np.asarray(extractedNumpyFromTrainData))

    iii += 1

