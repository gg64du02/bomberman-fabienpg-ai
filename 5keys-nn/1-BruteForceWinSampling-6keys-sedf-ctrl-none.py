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

# to generate alternative timing for keystroke to get around more easily
from numpy import random


# MoveToTheTileNextToMe(player1indexes,node)
# rightward (0,1)
# left (0,-1)
# downward (1,0)
# upward (-1,0)
def MoveToTheTileNextToMe(playerPos, nextStepPos):

    print("MoveToTheTileNextToMe:",playerPos, nextStepPos)
    # timePress = 0.15
    timePress = 0.10+random.randint(5)*0.01
    # timePress = 0.10+random.randint(10)*0.01
    # timePress = random.randint(5)*0.01

    # upward
    if(playerPos[0]>nextStepPos[0]):
        keyboard.press('e')
        time.sleep(timePress)
        keyboard.release('e')
    # downward
    if(playerPos[0]<nextStepPos[0]):
        keyboard.press('d')
        time.sleep(timePress)
        keyboard.release('d')
    # rightward
    if(playerPos[1]<nextStepPos[1]):
        keyboard.press('f')
        time.sleep(timePress)
        keyboard.release('f')
    # leftward
    if (playerPos[1] > nextStepPos[1]):
        keyboard.press('s')
        time.sleep(timePress)
        keyboard.release('s')

    pass


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
    file_name = './phase-1-bruteforce/training_data-{}.npy'.format(starting_value)
    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)
        break

def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    stop = False
    print('STARTING!!!')

    intI = 0

    while(True):
        print("========================")

        numberOfKeys = len(all_keys)

        print("numberOfKeys",numberOfKeys)

        roundEnded = False

        i = 0

        # screenshotTaken = []
        # keyIssued = []

        if(stop == True):
            break

        game_data = []

        while(roundEnded == False):

            choosedKey = random.randint(numberOfKeys)

            print("choosedKey",choosedKey)

            # getting the window mode screen
            screen = grab_screen(region=(anchorHWidthTopLeft, anchorHeightTopLeft,
                                         anchorWidthBotRight, anchorHeightBotRight))

            # pixels characters 2
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (int(WIDTH / 2), int(HEIGTH / 2)))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


            timePress = 0.05 + 0.01 * random.randint(7)

            # e =     [1,0,0,0,0,0]
            # d =     [0,1,0,0,0,0]
            # s =     [0,0,1,0,0,0]
            # f =     [0,0,0,1,0,0]
            # ctrl =  [0,0,0,0,1,0]
            # none =  [0,0,0,0,0,1]
            # all_keys = [e,d,s,f,ctrl,none]
            if(choosedKey == 0):
                keyboard.press('e')
                time.sleep(timePress)
                keyboard.release('e')
            if(choosedKey == 1):
                keyboard.press('d')
                time.sleep(timePress)
                keyboard.release('d')
            if(choosedKey == 2):
                keyboard.press('s')
                time.sleep(timePress)
                keyboard.release('s')
            if(choosedKey == 3):
                keyboard.press('f')
                time.sleep(timePress)
                keyboard.release('f')
            if(choosedKey == 4):
                keyboard.press('ctrl')
                time.sleep(timePress)
                keyboard.release('ctrl')
            if (choosedKey == 5):
                print("none")

            if (keyboard.is_pressed('n') == True):
                stop = True

                keyboard.release('ctrl')
                keyboard.release('e')
                keyboard.release('d')
                keyboard.release('s')
                keyboard.release('f')

                break

            # screenshotTaken.append(screen)
            # keyIssued.append(choosedKey)

            game_data.append([screen, choosedKey])


            # TODO: check for variable set to one in memory
            #  when the round is restarting/ending
            i+=1
            if(i == 500):
                roundEnded = True
                file_name = './phase-1-bruteforce/training_data-{}.npy'.format(starting_value)
                np.save(file_name, game_data)
                print('SAVED')
                starting_value += 1

main(file_name, starting_value)


