import numpy as np

import cv2
import time
import os

from grabscreen import grab_screen

import keyboard


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
    return (1,1)
    # pass

def AvailiablePath(screen,number):
    pass

def ScreenAveraging(screen):
    # tile of 32 piwel
    tileWidth = 32
    screenAveragedRet = np.zeros((20,10,3))
    tilePositons = tilePositionGenerator()
    for tilePos in tilePositons:
        # print(tilePos)
        tp = tilePos
        screenAveragedRet[int(tp[0]/tileWidth)][int(tp[1]/tileWidth)] = np.average(screen[tp[0]:tp[2],tp[1]:tp[3]])
    print("screenAveragedRet:",screenAveragedRet)

    pass

def tilePositionGenerator():
    # tile of 32 piwel
    tileWidth = 32
    for i in range(20):
        for ii in range(10):
            yield(i*tileWidth,ii*tileWidth,(i+1)*tileWidth,(ii+1)*tileWidth)

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

anchorHeightBotRight = anchorHeightTopLeft + HEIGTH
anchorWidthBotRight = anchorHWidthTopLeft + WIDTH



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
    # resize to something a bit more acceptable for a CNN

    # screen = cv2.resize(screen, (int(WIDTH / 2), int(HEIGTH / 2)))
    # screen = cv2.resize(screen, (int(WIDTH / 8), int(HEIGTH / 8)))

    # run a color convert:
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    print("GetPlayerPosition(1):",GetPlayerPosition(screen,1))

    availiablePath = AvailiablePath(screen, 1)


    screenAveraged = ScreenAveraging(screen)



    if (keyboard.is_pressed('p') == True):
        paused = True
        cv2.destroyAllWindows()
        break

    # too see what is captured
    cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF == ord('t'):
        cv2.destroyAllWindows()
        break
