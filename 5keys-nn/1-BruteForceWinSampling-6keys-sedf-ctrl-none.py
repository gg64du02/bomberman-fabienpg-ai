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

# ============================================================

SPEEDHACK_SPEED = 1

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

            # choosedKey = random.randint(numberOfKeys)
            choosedKey = random.randint(300000) % 6

            print("choosedKey",choosedKey)

            # getting the window mode screen
            screen = grab_screen(region=(anchorHWidthTopLeft, anchorHeightTopLeft,
                                         anchorWidthBotRight, anchorHeightBotRight))

            # pixels characters 2
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (int(WIDTH / 2), int(HEIGTH / 2)))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


            timePress = ( 0.05 + 0.01 * random.randint(7) ) / SPEEDHACK_SPEED

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

            # TODO: check when player2 is dead
            # TODO: check for variable set to one in memory
            #  when the round is restarting/ending

            # 1 if a players in the lasts seconds
            tmpOffset = int(0x4440B0)
            print("tmpOffset", '%s' % hex(tmpOffset))

            ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff, bufferSize, c.byref(bytesRead))
            aplayerDown = unpack('I', buff)[0]
            print("playerDown", aplayerDown)


            # 1 or >1 if a players in the lasts seconds
            tmpOffset = int(0x455C9C)
            print("tmpOffset", '%s' % hex(tmpOffset))

            ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff, bufferSize, c.byref(bytesRead))
            numbersOFDeathInLastSeconds = unpack('I', buff)[0]
            print("numbersOFDeathInLastSeconds", numbersOFDeathInLastSeconds)

            if(numbersOFDeathInLastSeconds==1):
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            else:
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


            # It finally read the string
            buff2 = c.create_string_buffer(32)
            bufferSize2 = (c.sizeof(buff2))
            print("bufferSize2",bufferSize2)
            bytesRead2 = c.c_ulonglong(0)
            tmpOffset = int(0x43FD49)
            print("tmpOffset", '%s' % hex(tmpOffset))

            ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff2, bufferSize2, c.byref(bytesRead2))
            scoreStr = unpack('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB', buff2)
            # 45 is -
            print("\n\nscoreStr", scoreStr)
            test3 = [chr(i) for i in scoreStr]
            test4 = ""
            test5 = test4.join(test3)
            print("test3",test3)
            print("test4",test4)
            print("test5",test5)
            test6 = test5.replace('\n',' ')
            print("test6",test6)
            test7 = test6.split(' ')
            print("test7",test7)
            p1score = int(test7[0])
            print("p1score",p1score)
            p1kill = int(test7[4])
            print("p1kill",p1kill)
            p1death = int( test7[5].split('/')[1] )
            print("p1death",p1death)

            print()



            # drop the current capture if p1 and p2 died
            if(numbersOFDeathInLastSeconds==1):
                roundEnded = True
                break

            # p1 and p2 are dead (including both killed themself)
            if(numbersOFDeathInLastSeconds>1):
                roundEnded = True
                file_name = './phase-1-bruteforce/training_data-{}.npy'.format(starting_value)
                np.save(file_name, game_data)
                print('SAVED')
                starting_value += 1

            # stop the recording if it is too long (and kill the player ?)
            if(i == 50000):
                roundEnded = True
                break



            i+=1
            # if(i == 500):
            #     roundEnded = True
            #     # file_name = './phase-1-bruteforce/training_data-{}.npy'.format(starting_value)
            #     # np.save(file_name, game_data)
            #     # print('SAVED')
            #     # starting_value += 1

main(file_name, starting_value)


