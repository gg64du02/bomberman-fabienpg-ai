import numpy as np
import cv2
import time
import pyautogui
from grabscreen import grab_screen
# from keys import Keys
#import playsound
import keyboard
# import key_pressed_and_issuing
import os
# for checking free space
import ctypes
import platform

# to generate alternative timing for keystroke to get around more easily
from numpy import random

# for memory reading
import ctypes as c
from ctypes import wintypes as w
from struct import *
from time import *
import datetime
import sys
import time

# getting PID
import psutil

# for debbuging purposes on long runs
import traceback


# ============================================
e =     [1,0,0,0,0,0]
d =     [0,1,0,0,0,0]
s =     [0,0,1,0,0,0]
f =     [0,0,0,1,0,0]
ctrl =  [0,0,0,0,1,0]
none =  [0,0,0,0,0,1]
all_keys = [e,d,s,f,ctrl,none]

StopNotPressed = False

# your desktop resolution:
desktopHeight = 1080
desktopWidth = 1920

# 640*480
HEIGTH = 480
WIDTH = 640

intSubdiv = 24
#bugfix missing last lines
iHEIGTH = (HEIGTH//intSubdiv)
iWIDTH = (WIDTH//intSubdiv)
#bugfix inverted width height

# calculating top left anchor point in window mode:
anchorHeightTopLeft = int( ( desktopHeight - HEIGTH ) / 2 )
anchorHWidthTopLeft = int( ( desktopWidth - WIDTH ) / 2 )

anchorHeightBotRight = anchorHeightTopLeft + HEIGTH
anchorWidthBotRight = anchorHWidthTopLeft + WIDTH


if 0:
    print("desktopHeight:"+str(desktopHeight))
    print("desktopWidth:"+str(desktopWidth))

    print("anchorHeightTopLeft:"+str(anchorHeightTopLeft))
    print("anchorHWidthTopLeft:"+str(anchorHWidthTopLeft))

    print("anchorHeightBotRight:"+str(anchorHeightBotRight))
    print("anchorWidthBotRight:"+str(anchorWidthBotRight))


starting_value = 1
while True:
    file_name = './phase-1-builtin-ai/training_data-{}.npy'.format(starting_value)
    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)
        break

# ==================MEMORY READING PREP======================
# TODO: please set this or automate it

# get last Bomber.exe PID
PROCNAME = "Bomber.exe"

pid = -1

for proc in psutil.process_iter():
    if proc.name() == PROCNAME:
        print(proc)
        pid = proc.pid
        # pid = 1960
if(pid<0):
    exit()
print("pid",pid)

k32 = c.windll.kernel32

OpenProcess = k32.OpenProcess
OpenProcess.argtypes = [w.DWORD,w.BOOL,w.DWORD]
OpenProcess.restype = w.HANDLE

ReadProcessMemory = k32.ReadProcessMemory
ReadProcessMemory.argtypes = [w.HANDLE,w.LPCVOID,w.LPVOID,c.c_size_t,c.POINTER(c.c_size_t)]
ReadProcessMemory.restype = w.BOOL
PAA = 0x1F0FFF
# PAA = 0x19F7D0
startAddress = 0x0000000
# startAddress = 0x4000000
endAddress = 0x5000000

ph = OpenProcess(PAA,False,int(pid)) #program handle

buff = c.create_string_buffer(4)
bufferSize = (c.sizeof(buff))
bytesRead = c.c_ulonglong(0)

listOf680 = []

# 0x4 : 4 bytes steps
# addresses_list = xrange(address,0x9000000,0x4)
addresses_list = range(startAddress,endAddress,0x4)

def get_free_space_mb(dirname):
    """Return folder/drive free space (in megabytes)."""
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
    else:
        st = os.statvfs(dirname)
        return st.f_bavail * st.f_frsize / 1024 / 1024

# ============================================================

def getScoreKillsDeaths():

    # It finally read the string
    buff2 = c.create_string_buffer(32)
    bufferSize2 = (c.sizeof(buff2))
    # print("bufferSize2", bufferSize2)
    bytesRead2 = c.c_ulonglong(0)
    # 44FD3E
    tmpOffset = int(0x43FD49)
    # print("tmpOffset", '%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
    scoreStr = unpack('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB', buff2)
    # 45 is -
    # print("\n\nscoreStr", scoreStr)
    test3 = [chr(i) for i in scoreStr]
    test4 = ""
    test5 = test4.join(test3)
    # print("test3", test3)
    # print("test4", test4)
    # print("test5", test5)
    test6 = test5.replace('\n', ' ')
    print("test6", test6)
    test7 = test6.split(' ')
    print("test7", test7)
    print("test7[0]",test7[0])
    try:
        p1score = int(test7[1])
        p1kill = int(test7[5])
        p1death = int(test7[5].split('/')[0])
    except:
        p1score = int(test7[0])
        p1kill = int(test7[4])
        p1death = int(test7[5].split('/')[1])

    print("p1score", p1score)
    print("p1kill", p1kill)
    print("p1death", p1death)

    return p1score,p1kill,p1death


def getScoreKillsDeaths2():

    # It finally read the string
    buff2 = c.create_string_buffer(96)
    bufferSize2 = (c.sizeof(buff2))
    # print("bufferSize2", bufferSize2)
    bytesRead2 = c.c_ulonglong(0)
    # 44FD3E
    # tmpOffset = int(0x43FD40)
    tmpOffset = int(0x0044FD20)
    # print("tmpOffset", '%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
    scoreStr = unpack('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB', buff2)
    # 45 is -
    # print("\n\nscoreStr", scoreStr)
    test3 = [chr(i) for i in scoreStr]
    test4 = ""
    test5 = test4.join(test3)
    # print("test3", test3)
    # print("test4", test4)
    # print("test5", test5)
    test6 = test5.replace('\n', ' ')
    # print("test6", test6)
    test7 = test6.split(' ')
    print("test7", test7)
    print("test7[0]",test7[0])
    # TODO: the problem spawn everytime you check/uncheck the AI
    # test7 = ['Joueur', '2', '(IA)', '-1', 'Points', '', '', '0', 'K/1', 'AK', '', 'Joueur', '1', '(IA)', '-2', 'Points', '', '', '0', '']

    offset = 0

    # checking if it is an AI
    if(test7[2]=='(IA)'):
        offset+=1
        print("first list is controlled by AI")
    else:
        print("first list is controlled by human")

    # if the player 1 is listed as first
    if(test7[1]=='1'):
        p1score = int(test7[2 + offset])
        p1kill = int(test7[6 + offset])
        p1death = int(test7[6 + offset].split('/')[0])
    else:
        # checking if it is an AI
        if (test7[12+offset] == '(IA)'):
            offset += 1
            print("second listed is controlled by AI")
        else:
            print("second listed is controlled by human")

        p1score = int(test7[12+offset])
        p1kill = int(test7[16+offset])
        p1death = int(test7[17+offset].split('/')[1])

        # if the player 2 is listed as first
        if(test7[10+offset] == '1'):
            print("if(test7[10+offset] == '1'):")

        else:
            print('error nothing found')

    # for debbugging purpose
    # p1score = 0
    # p1kill = 0
    # p1death = 0

    print("p1score", p1score)
    print("p1kill", p1kill)
    print("p1death", p1death)

    return p1score,p1kill,p1death

def isRightMove():
    # RIGHT
    # It finally read the string
    buff2 = c.create_string_buffer(1)
    bufferSize2 = (c.sizeof(buff2))
    # print("bufferSize2", bufferSize2)
    bytesRead2 = c.c_ulonglong(0)
    tmpOffset = int(0x00455B50)
    # print("tmpOffset", '%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
    scoreStr = unpack('B', buff2)
    # 45 is -
    # print("\n\nscoreStr", scoreStr)
    test3 = [int(i) for i in scoreStr]
    # print("test3",test3)
    # debugging purpose
    if(test3[0] == 6):
        return 'right'
    return ''

def isLeftMove():
    # LEFT
    # It finally read the string
    buff2 = c.create_string_buffer(1)
    bufferSize2 = (c.sizeof(buff2))
    # print("bufferSize2", bufferSize2)
    bytesRead2 = c.c_ulonglong(0)
    tmpOffset = int(0x00455B51)
    # print("tmpOffset", '%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
    scoreStr = unpack('B', buff2)
    # 45 is -
    # print("\n\nscoreStr", scoreStr)
    test3 = [int(i) for i in scoreStr]
    # print("test3",test3)
    # debugging purpose
    if(test3[0] == 255):
        return 'left'
    return ''

def isUpMove():
    # UP
    # It finally read the string
    buff2 = c.create_string_buffer(1)
    bufferSize2 = (c.sizeof(buff2))
    # print("bufferSize2", bufferSize2)
    bytesRead2 = c.c_ulonglong(0)
    tmpOffset = int(0x00455B54)
    # print("tmpOffset", '%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
    scoreStr = unpack('B', buff2)
    # 45 is -
    # print("\n\nscoreStr", scoreStr)
    test3 = [int(i) for i in scoreStr]
    # print("test3",test3)
    # debugging purpose
    if(test3[0] == 250):
        return 'up'
    return ''

def isDownMove():
    # DOWN
    # It finally read the string
    buff2 = c.create_string_buffer(1)
    bufferSize2 = (c.sizeof(buff2))
    # print("bufferSize2", bufferSize2)
    bytesRead2 = c.c_ulonglong(0)
    tmpOffset = int(0x00455B54)
    # print("tmpOffset", '%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
    scoreStr = unpack('B', buff2)
    # 45 is -
    # print("\n\nscoreStr", scoreStr)
    test3 = [int(i) for i in scoreStr]
    # print("test3",test3)
    # debugging purpose
    if(test3[0] == 6):
        return 'down'
    return ''

def numberOfRemainingBombs():
    # BOMBS
    # It finally read the string
    buff2 = c.create_string_buffer(1)
    bufferSize2 = (c.sizeof(buff2))
    # print("bufferSize2", bufferSize2)
    bytesRead2 = c.c_ulonglong(0)
    tmpOffset = int(0x00455AFA)
    # print("tmpOffset", '%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
    scoreStr = unpack('B', buff2)
    # 45 is -
    # print("\n\nscoreStr", scoreStr)
    test3 = [int(i) for i in scoreStr]
    # print("test3",test3)
    # debugging purpose
    return test3[0]

def getArrowDirection():
    # at 00455AFA is the number of remaining bombs on one byte
    # at 00455B54 is 06 would be down  on one byte ok
    # at 00455B54 is FA would be up    on one byte ok
    # at 00455B51 is FF would be left  on one byte ok
    # at 00455B50 is 06 would be right on one byte ok
    strTemp = ''

    isUp = isUpMove()
    if(isUp!=''):
        return 'up'

    isDown = isDownMove()
    if(isDown!=''):
        return 'down'

    isLeft= isLeftMove()
    if(isLeft!=''):
        return 'left'

    isRight = isRightMove()
    if(isRight!=''):
        return 'right'

    return 'none'


# Speedhack speed fixed with Cheat Engine
SPEEDHACK_SPEED = 1
print("SPEEDHACK_SPEED",SPEEDHACK_SPEED)

DEBUG = 0

def main(file_name, starting_value):

    keyboard.press('tab')
    time.sleep(0.02/SPEEDHACK_SPEED)
    keyboard.release('tab')
    time.sleep(0.02/SPEEDHACK_SPEED)

    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    previousnumberOfRemainingBombs = 0

    stop = False
    print('STARTING!!!')

    intI = 0

    # supposed to be at least not a dead player at the start
    numbersOFDeathInLastSeconds = 0

    # p1scoreAS, p1killAS, p1deathAS = getScoreKillsDeaths()
    p1scoreAS, p1killAS, p1deathAS = getScoreKillsDeaths2()

    # AS: At Start
    print("at start p1scoreAS, p1killAS, p1deathAS", p1scoreAS, p1killAS, p1deathAS)

    while(True):
        print("========================")

        numberOfKeys = len(all_keys)

        print("numberOfKeys",numberOfKeys)

        roundEnded = False

        i = 0

        screenshotTaken = []
        previousScreenshotTaken = []
        keyIssued = []

        if(stop == True):
            break

        if(starting_value > 1000):
            print("if(starting_value > 1000):")
            exit()

        game_data = []

        print("waiting for a new round")

        while(numbersOFDeathInLastSeconds==1):
            # 1 or >1 if a players in the lasts seconds
            tmpOffset = int(0x455C9C)
            # print("tmpOffset", '%s' % hex(tmpOffset))

            ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff, bufferSize, c.byref(bytesRead))
            numbersOFDeathInLastSeconds = unpack('I', buff)[0]
            # print("numbersOFDeathInLastSeconds", numbersOFDeathInLastSeconds)

        print("new round started")

        while(roundEnded == False):

            bombsIssuedOrNot = False

            # # choosedKey = random.randint(numberOfKeys)
            # choosedKey = random.randint(300000) % 6
            #
            # print("choosedKey",choosedKey)

            # getting the window mode screen
            screen = grab_screen(region=(anchorHWidthTopLeft, anchorHeightTopLeft,
                                         anchorWidthBotRight, anchorHeightBotRight))

            # pixels characters 2
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (int(WIDTH / 2), int(HEIGTH / 2)))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


            # at 00455AFA is the number of remaining bombs on one byte
            # at 00455B54 is 06 would be down  on one byte ok
            # at 00455B54 is FA would be up    on one byte ok
            # at 00455B51 is FF would be left  on one byte ok
            # at 00455B50 is 06 would be right on one byte ok
            # none

            # TODO: implement here
            arrowsPressedOrNot = getArrowDirection()
            if(DEBUG):
                print("arrowsPressedOrNot",arrowsPressedOrNot)

            numbersOfRemainingBombsToPlace = numberOfRemainingBombs()
            # print("numbersOfRemainingBombsToPlace",numbersOfRemainingBombsToPlace)

            # DONE:bugfix
            if(previousnumberOfRemainingBombs > numbersOfRemainingBombsToPlace ):
                print("bombIssued")
                bombsIssuedOrNot = True

            previousnumberOfRemainingBombs = numbersOfRemainingBombsToPlace

            if(bombsIssuedOrNot==True):
                # bomb
                choosedKey = [0,0,0,0,1,0]
            else:
                if(arrowsPressedOrNot=='up'):
                    choosedKey = [1,0,0,0,0,0]
                if(arrowsPressedOrNot=='down'):
                    choosedKey = [0,1,0,0,0,0]
                if(arrowsPressedOrNot=='left'):
                    choosedKey = [0,0,1,0,0,0]
                if(arrowsPressedOrNot=='right'):
                    choosedKey = [0,0,0,1,0,0]
                if(arrowsPressedOrNot=='none'):
                    choosedKey = [0,0,0,0,0,1]
                if(arrowsPressedOrNot==''):
                    choosedKey = [0,0,0,0,0,1]

            # screenshotTaken.append(screen)
            # game_data.append([screen, choosedKey]

            if(previousScreenshotTaken!=[]):
                # screenshotTaken.append(previousScreenshotTaken)
                # keyIssued.append(choosedKey)

                game_data.append([previousScreenshotTaken, choosedKey])

            previousScreenshotTaken = screen

            # TODO: check when player2 is dead
            # TODO: check for variable set to one in memory
            #  when the round is restarting/ending

            # 1 if a players in the lasts seconds
            tmpOffset = int(0x4440B0)
            # print("tmpOffset", '%s' % hex(tmpOffset))

            ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff, bufferSize, c.byref(bytesRead))
            aplayerDown = unpack('I', buff)[0]
            # print("playerDown", aplayerDown)


            # 1 or >1 if a players in the lasts seconds
            tmpOffset = int(0x455C9C)
            # print("tmpOffset", '%s' % hex(tmpOffset))

            ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff, bufferSize, c.byref(bytesRead))
            numbersOFDeathInLastSeconds = unpack('I', buff)[0]
            if(DEBUG):
                print("numbersOFDeathInLastSeconds", numbersOFDeathInLastSeconds)



            if(numbersOFDeathInLastSeconds!=0):

                keyboard.press('tab')
                time.sleep(0.02 / SPEEDHACK_SPEED)
                keyboard.release('tab')
                time.sleep(0.02 / SPEEDHACK_SPEED)

                # p1scoreNew, p1killNew, p1deathNew = getScoreKillsDeaths()
                p1scoreNew, p1killNew, p1deathNew = getScoreKillsDeaths2()

                print("p1scoreNew, p1killNew, p1deathNew", p1scoreNew, p1killNew, p1deathNew)


            # drop the current capture if p1 and p2 died
            if(numbersOFDeathInLastSeconds==1):
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                roundEnded = True

                print("sanity check",p1killNew,p1killAS)

                # Mean the controlled player kill count changed
                # if(p1killNew!=p1killAS):
                if(p1killNew>p1killAS):

                    file_name = './phase-1-builtin-ai/training_data-{}.npy'.format(starting_value)

                    # shuffle
                    print("random.shuffle(game_data)")
                    random.shuffle(game_data)

                    # =======================
                    # lenghtInt = len((game_data[:, 1])[0])
                    lenghtInt = len(game_data[0][1])

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
                        print("Outputs:", Outputs)
                        i6 = 0
                        # test_argwhere = np.argwhere((train_data[:, 1])[i6]!=0)
                        # print()

                        while (i6 < len(game_data)):
                            # tmpArray = np.asarray((game_data[:, 1])[i6])
                            tmpArray = np.asarray(game_data[i6][1])
                            tmpArray2 = np.asarray(Outputs)
                            # print("tmpArray:",tmpArray)
                            # print("tmpArray2:",tmpArray2)
                            if (np.array_equal(tmpArray, tmpArray2)):
                                # print("np.array_equal(tmpArray,tmpArray2)")
                                # print("i6:",i6)
                                # print("Outputs:",Outputs)
                                outputsOccurence[outputs_number].append([i6])
                            i6 += 1
                        print("len(outputsOccurence[" + str(outputs_number) + "]):",
                              len(outputsOccurence[outputs_number]))

                        NumberOfOccurenceOutput.append(len(outputsOccurence[outputs_number]))

                        outputs_number += 1

                    # bare minimum to extract
                    minimumToExtract = min(NumberOfOccurenceOutput)
                    # preparing the numpy for file writing
                    # extractedNumpyFromTrainData = np.zeros((lenghtInt*minimumToExtract,2))
                    extractedNumpyFromTrainData = [[[], []] for line in range(int(lenghtInt * minimumToExtract))]
                    # extractedNumpyFromTrainData = np.zeros((int(lenghtInt*minimumToExtract),2))

                    print("len(extractedNumpyFromTrainData):", len(extractedNumpyFromTrainData))

                    # counting remaining
                    outputsOccurenceInOutputFile = NumberOfOccurenceOutput

                    # outputsOccurence[outputs_number])
                    print()
                    i7 = 0

                    i9 = 0
                    print("len(outputsOccurence):", len(outputsOccurence))
                    while (i7 < len(outputsOccurence)):
                        # print(outputsOccurence[i7])
                        print("i7:", i7)
                        i8 = 0
                        # lol (outputsOccurence[5])[0]
                        while (i8 < minimumToExtract):
                            print("i8:", i8)
                            # print((outputsOccurence[i7])[i8])
                            # print( ( (outputsOccurence[i7])[i8] )[0]      )
                            tmpIndex = ((outputsOccurence[i7])[i8])[0]
                            print("tmpIndex:", tmpIndex)

                            extractedNumpyFromTrainData[i9] = game_data[tmpIndex]

                            print("i9:", i9)

                            # index for tain_data
                            i9 += 1

                            # iteration on each occurence
                            i8 += 1

                        # iteration on another output
                        i7 += 1

                    print()

                    # file_name_partial_extraction_dataset = './phase-3/training_data_merged-partial-dataset-{}.npy'.format(
                    #     iii)

                    # np.save(file_name_partial_extraction_dataset, extractedNumpyFromTrainData)

                    # extractedNumpyFromTrainData is a list
                    # np.save(file_name_partial_extraction_dataset, np.asarray(extractedNumpyFromTrainData))

                    try:
                        extractedNumpyFromTrainDataTmp = [[] for i in range(len(extractedNumpyFromTrainData))]

                        offset = int(len(extractedNumpyFromTrainData) / len(extractedNumpyFromTrainData[0][1]))
                        lineNum = 0

                        for indexOffset in range(offset):
                            for outputKind in range(len(extractedNumpyFromTrainData[0][1])):
                                index = int(offset * outputKind) + indexOffset
                                extractedNumpyFromTrainDataTmp[lineNum] = np.asarray(extractedNumpyFromTrainData)[index]
                                lineNum += 1

                        # print()

                        # extractedNumpyFromTrainData is a list
                        # np.save(file_name_partial_extraction_dataset, np.asarray(extractedNumpyFromTrainDataTmp))
                        np.save(file_name, np.asarray(extractedNumpyFromTrainDataTmp))

                        # =======================

                        # np.save(file_name, game_data)
                        print('SAVED in builtin-ai folder')
                    except:
                        print("avoiding script's crash")

                        # chack https://docs.python.org/3.1/library/traceback.html#traceback-examples
                        print("Exception in user code:")
                        print("-" * 60)
                        traceback.print_exc(file=sys.stdout)
                        print("-" * 60)
                        starting_value -= 1

                    starting_value += 1

                    keyboard.press('tab')
                    time.sleep(0.02/SPEEDHACK_SPEED)
                    keyboard.release('tab')
                    time.sleep(0.02/SPEEDHACK_SPEED)

                    # updating infos about the game state, a break is required to work properly
                    p1killAS = p1killNew
                    p1kdeathAS = p1deathNew
                    p1scoreAS = p1killNew

                    #  115646.1171875 Megabytes
                    if((get_free_space_mb('d:/') < 20000)or(get_free_space_mb('c:/') < 40000)):
                        print("stopping generation","if((get_free_space_mb('d:/') < 20000)or(get_free_space_mb('c:/') < 40000)):")
                        exit()
                    else:
                        print("continuing generation","!if((get_free_space_mb('d:/') < 20000)or(get_free_space_mb('c:/') < 40000)):")

                    pass

                break
            else:
                pass
                # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            # # p1 and p2 are dead (including both killed themself)
            # if(numbersOFDeathInLastSeconds>1):
            #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #     roundEnded = True
            #     file_name = './phase-1-bruteforce/training_data-{}.npy'.format(starting_value)
            #
            #     # implement here
            #     # =============================================================================
            #     # =============================================================================
            #
            #     # np.save(file_name, game_data)
            #     print('SAVED in bruteforce folder')
            #     starting_value += 1
            #
            #     keyboard.press('tab')
            #     time.sleep(0.02/SPEEDHACK_SPEED)
            #     keyboard.release('tab')
            #     time.sleep(0.02/SPEEDHACK_SPEED)
            #
            #     # updating infos about the game state, a break is required to work properly
            #     p1killAS = p1killNew
            #     p1kdeathAS = p1deathNew
            #     p1scoreAS = p1killNew
            #
            #     break


            # stop the recording if it is too long (and kill the player ?)
            if(i == 10000):
                roundEnded = True
                break


            # print("i",i)
            i+=1


main(file_name, starting_value)


