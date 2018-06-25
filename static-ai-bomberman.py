import numpy as np

import cv2
import time
import os

from grabscreen import grab_screen

import keyboard

# to process to potential centroid for each player (ie player positions
from scipy import ndimage


def IssueKeystroke(key):
    keyboard.press(key)
    time.sleep(0.05)
    keyboard.release(key)

def MapMaskGenerator():
    pass

def GoToPosition(X,Y):
    pass

def GetPlayerPosition(screen, number):
    # number 1 2 3 4

    H_min = 0
    S_min = 60
    V_min = 120
    H_max = 10
    S_max = 255
    V_max = 255
    frame_to_thresh = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(
        frame_to_thresh, (H_min, S_min, V_min), (H_max, S_max, V_max))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # print(cnts)

    sum_x = 0
    sum_y = 0

    for i in cnts:
        # print("i:",i)
        for ii in i:
            # print("ii[0]:",ii[0])
            sum_x += ii[0,0]
            sum_y += ii[0,1]
        sum_x = sum_x / len(i)
        sum_y = sum_y / len(i)
    # print(sum_x,sum_y)

    return (sum_x,sum_y)

    # pos3D = ndimage.measurements.center_of_mass(np.asarray(cnts))
    #
    # return pos3D

def AvailiablePath(screen,screenAveraged,number):
    crate = [65,151,191]
    hardBlock = [156,156,156]
    bomb = [46,56,58]

    allBlocking = [crate,hardBlock,bomb]

    availiableSpots = np.zeros((20,10))

    tp = tilePositionGenerator()
    for tile in tp:
        for i in allBlocking:
            print()
            if(np.array_equal(i,tile)==True):
                availiableSpots[int(tile[0]/32),int(tile[1]/32)] = False
            else:
                availiableSpots[int(tile[0]/32),int(tile[1]/32)] = True
    return availiableSpots

def ScreenAveraging(screen):

    # print("screen.shape:",screen.shape)

    # tile of 32 piwel
    tileWidth = 32
    screenAveragedRet = np.zeros((10,20,3))
    tilePositons = tilePositionGenerator()
    for tilePos in tilePositons:
        # print(tilePos)
        tp = tilePos

        screenAveragedRet[int(tp[1]/tileWidth),int(tp[0]/tileWidth),0] = np.average(screen[tp[1]:tp[3],tp[0]:tp[2],0])
        screenAveragedRet[int(tp[1]/tileWidth),int(tp[0]/tileWidth),1] = np.average(screen[tp[1]:tp[3],tp[0]:tp[2],1])
        screenAveragedRet[int(tp[1]/tileWidth),int(tp[0]/tileWidth),2] = np.average(screen[tp[1]:tp[3],tp[0]:tp[2],2])
        # print("tilePos:",tilePos)
    # print("screenAveragedRet:",screenAveragedRet)
    return screenAveragedRet

def tilePositionGenerator():
    # tile of 32 piwel
    tileWidth = 32
    for i in range(20):
        for ii in range(10):
            # print(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)
            yield(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)

    pass

def MapMaskGenerator():
    pass

# your desktop resolution:
desktopHeight = 1080
desktopWidth = 1920

# 640*480
HEIGTH = 480
WIDTH = 640


# calculating top left anchor point in window mode:
anchorHeightTopLeft = int( ( desktopHeight - HEIGTH ) / 2 )
anchorHWidthTopLeft = int( ( desktopWidth - WIDTH ) / 2 )

anchorHeightBotRight = anchorHeightTopLeft + HEIGTH - 1
anchorWidthBotRight = anchorHWidthTopLeft + WIDTH - 1



# def

stop = False

if 0:
    tilePositons = tilePositionGenerator()
    for tilePos in tilePositons:
        print(tilePos)
    print()
    # break
while True:

    # getting the window mode screen
    screen = grab_screen(region=(anchorHWidthTopLeft, anchorHeightTopLeft,
                                 anchorWidthBotRight, anchorHeightBotRight))

    # pixels characters 2
    # screen = cv2.resize(screen, (int(WIDTH / 2), int(HEIGTH / 2)))
    # screen = cv2.resize(screen, (int(WIDTH / 8), int(HEIGTH / 8)))

    # run a color convert:
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    getPlayerPosition = GetPlayerPosition(screen, 1)

    # print("getPlayerPosition:",getPlayerPosition)

    screenAveraged = ScreenAveraging(screen)

    # availiablePath = AvailiablePath(screen,screenAveraged, 1)
    #
    # print(availiablePath)


    if (keyboard.is_pressed('p') == True):
        paused = True
        cv2.destroyAllWindows()
        break

    # too see what is captured
    # cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    # cv2.imshow('thresh', thresh)

    if cv2.waitKey(25) & 0xFF == ord('t'):
        cv2.destroyAllWindows()
        break
