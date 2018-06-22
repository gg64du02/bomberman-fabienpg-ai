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

# ============================================
# default player keyboard binding
# e =     [1,0,0,0,0,0]
# d =     [0,1,0,0,0,0]
# s =     [0,0,1,0,0,0]
# f =     [0,0,0,1,0,0]
# ctrl =  [0,0,0,0,1,0]
# shift = [0,0,0,0,0,1]
# all_keys = [e,d,s,f,ctrl,shift]

e =     [1,0,0,0,0]
d =     [0,1,0,0,0]
s =     [0,0,1,0,0]
f =     [0,0,0,1,0]
ctrl =  [0,0,0,0,1]
all_keys = [e,d,s,f,ctrl]

def what_keys_is_pressed():
    if(keyboard.is_pressed('e')):
        return e
    if(keyboard.is_pressed('d')):
        return d
    if(keyboard.is_pressed('s')):
        return s
    if(keyboard.is_pressed('f')):
        return f

    if(keyboard.is_pressed('ctrl')):
        return ctrl
    # if(keyboard.is_pressed('shift')):
    #     return shift
    return -1
# ============================================

# for i in list(range(4))[::-1]:
#     time.sleep(1)
#     print(str(i))

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
    file_name = './phase-1/training_data-{}.npy'.format(starting_value)
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

    last_time = time.time()
    paused = False
    print('STARTING!!!')

    intI = 0

    while(True):
        if(paused == True):
            time.sleep(1)
            paused = False
        # DONE: find the proper anchor
        while paused==False:
            print("========================")

            # print("another frame")

            last_time = time.time()

            if (keyboard.is_pressed('n') == True):
                paused = True

            # getting the window mode screen
            screen = grab_screen(region=(anchorHWidthTopLeft,anchorHeightTopLeft,
                                         anchorWidthBotRight,anchorHeightBotRight))

            last_time = time.time()

            # pixels characters 2
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (int(WIDTH/2), int(HEIGTH/2)))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            # which key is pressed
            key = what_keys_is_pressed()
            print("key:"+str(key))
            output = key
            if(key==-1):
                print("if(key==-1):")
            else:
                if(1):
                    intI = intI + 1
                    print("intI:"+str(intI))
                    training_data.append([screen,output])

                    if len(training_data) % 100 == 0:
                        print("len(training_data):" + str(len(training_data)))

                        if len(training_data) == 500:
                            np.save(file_name, training_data)
                            print('SAVED')
                            training_data = []
                            starting_value += 1
                            file_name = './phase-1/training_data-{}.npy'.format(starting_value)

            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()



            # print(type(screen))
            # too see what is captured
            cv2.imshow('screen',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(25) & 0xFF == ord('t'):
                cv2.destroyAllWindows()
                break





main(file_name, starting_value)








