import numpy as np

import cv2
import time
import os

from grabscreen import grab_screen

import keyboard

from skimage import measure

from scipy.spatial import distance

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

def convertToIndexesGetPlayerPosition(getPlayerPosition):
    playerXindex = int( ( getPlayerPosition[0] - getPlayerPosition[0] % 32 )/32 )
    playerYindex = int( ( getPlayerPosition[1] - getPlayerPosition[1] % 32 )/32 )
    return (playerXindex,playerYindex)


def availiablePathToControlledPlayer(availiablePath, getPlayerPosition):
    # playerXindex = int( ( getPlayerPosition[0] - getPlayerPosition[0] % 32 )/32 )
    # playerYindex = int( ( getPlayerPosition[1] - getPlayerPosition[1] % 32 )/32 )
    playerIndexPos = convertToIndexesGetPlayerPosition(getPlayerPosition)
    playerXindex = playerIndexPos[0]
    playerYindex = playerIndexPos[1]

    print(playerXindex,playerYindex)

    labeled = measure.label(availiablePath, background=False, connectivity=1)
    # reversed X,Y why ?
    label = labeled[playerYindex, playerXindex]  # known pixel location

    rp = measure.regionprops(labeled)
    props = rp[label - 1]  # background is labeled 0, not in rp

    # props.bbox  # (min_row, min_col, max_row, max_col)
    # props.image  # array matching the bbox sub-image
    print(len(props.coords))  # list of (row,col) pixel indices

    availiablePathRet = np.zeros((16,20))

    connectedCoords = props.coords
    for coord in connectedCoords:
        availiablePathRet[coord[0],coord[1]] = 1

    return connectedCoords,availiablePathRet

# support player 1, 2
def GetPlayerPosition(screen, number):
    # number 1 2 3 4

    if(number==1):
        H_min = 0
        S_min = 60
        V_min = 120
        H_max = 10
        S_max = 255
        V_max = 255
    if(number==2):
        # 152 100 50
        # 120 100 50
        # 120 66 78
        H_min = 56
        S_min = 30
        V_min = 125
        H_max = 136
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

# DONE: fix the "filled" (even though clear) bottom path issue
def AvailiablePath(screen,screenAveraged,number):
    crate = [65,151,191]
    hardBlock = [156,156,156]
    bomb = [46,56,58]

    screenAveragedToInt = screenAveraged.astype(int)

    allBlocking = [crate,hardBlock,bomb]

    availiableSpots = np.ones((16,20))

    tp = tilePositionGenerator()
    for tile in tp:
        for i in allBlocking:
            x = int(tile[0]/32)
            y = int(tile[1]/32)
            # print(x,y)
            # print("i:",i)
            # print("screenAveragedToInt[y,x]:",screenAveragedToInt[y,x])
            if(np.array_equal(i,screenAveragedToInt[y,x])==True):
                # print("True\n")
                availiableSpots[y,x] = False
            # else:
            #     print("!True\n")
    return availiableSpots

def ScreenAveraging(screen):

    # print("screen.shape:",screen.shape)

    # tile of 32 piwel
    tileWidth = 32
    screenAveragedRet = np.zeros((16,20,3))
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
        for ii in range(16):
            # print(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)
            yield(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)


# def closest_node(node, nodes):
#     closest_index = distance.cdist([node], nodes).argmin()
#     return nodes[closest_index]

def oneStepToPutBomb(potentialPath,potentialPathList,player1indexes,player2indexes):
    # a = random.randint(1000, size=(50000, 2))
    # some_pt = (1, 2)
    # print()
    # print("closest_node:",closest_node(player2indexes,potentialPathList))

    p2i = player2indexes

    # anynumber sup to 16**2 + 20**2
    curDir = 50

    for potPathElement in potentialPathList:
        x = potPathElement[0]
        y = potPathElement[1]
        # print("oneStepToPutBomb:",x,y)
        tmpDist = np.sqrt((x-p2i[0])**2+(y-p2i[1])**2)
        # print("tmpDist:",tmpDist)
        if(curDir>tmpDist):
            closest_node = (x,y)
            curDir=tmpDist


    print("closest_node:",closest_node)

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

    print()
    print("getPlayerPosition:",getPlayerPosition)

    screenAveraged = ScreenAveraging(screen)

    availiablePath = AvailiablePath(screen,screenAveraged, 1)

    # print(availiablePath)

    potentialPathList,potentialPath = availiablePathToControlledPlayer(availiablePath, getPlayerPosition)

    # TODO: remove the bottom line
    print(potentialPath)

    player1indexes = convertToIndexesGetPlayerPosition(GetPlayerPosition(screen,1))
    player2indexes = convertToIndexesGetPlayerPosition(GetPlayerPosition(screen,2))

    print("player1indexes:",player1indexes)
    print("player2indexes:",player2indexes)

    oneStepToPutBomb(potentialPath,potentialPathList,player1indexes,player2indexes)


    time.sleep(2)


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
