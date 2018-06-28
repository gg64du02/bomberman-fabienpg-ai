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

# to generate alternative timing for keystroke to get around more easily
from numpy import random

# sort and uniq
import itertools

# for a star
from heapq import *
import time

def MapMaskGenerator():
    pass

# Author: Christian Careaga (christian.careaga7@gmail.com)
# A* Pathfinding in Python (2.7)
# Please give credit if used

# credits:http://code.activestate.com/recipes/578919-python-a-pathfinding-with-binary-heap/
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):
    # neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    print("(start, goal):",(start, goal))

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))

    return False


# TODO: implement this
def isThereABombAtThisPosition(position):

    pass

# TODO: fixes thresholds
def listBombsPositions(screenAvged):
    # print("listBombsPostions")
    tp = tilePositionGenerator()
    list = []
    for potentialBombs in tp:
        x = int( potentialBombs[0] / 32 )
        y = int( potentialBombs[1] / 32 )

        # for debugging
        # if(x==0):
        #     if(y==0):
        #         print("potentialBombs:",potentialBombs)
        #         print("screenAveraged[y,x]:",screenAveraged[y,x])

        # [45.29032258 55.72528616 62.90114464]
        # [49.06971904 60.13735692 61.8855359]
        # [48.4037461  58.98855359 60.85327784]
        # [48.08740895 58.87200832 60.35379813]
        # [47.421436  57.3735692 59.2049948]
        # [46.27263267 56.09157128 57.52341311]
        # [46.15608741 54.89281998 56.05827263]
        # [45.29032258 53.87721124 54.82622268]
        if(screenAveraged[y,x,0]<=50):
            if(screenAveraged[y,x,0]>45):
                if(screenAveraged[y,x,1]<=61):
                    if(screenAveraged[y,x,1]>53):
                        if(screenAveraged[y,x,2]<=63):
                            if(screenAveraged[y,x,2]>54):
                                # print("bomb decteted")
                                list.append([y,x])

    return list


def isIndexesRange(point):
    isInsideIndexRange = False
    if (point[0] >= 0):
        if (point[0] <= 19):
            if (point[1] >= 0):
                if (point[1] <= 15):
                    # print("ii[0] in [0-19]")
                    # print("ii[1] in [0-15]")
                    print("ii in (0,0) and (19,15)")
                    isInsideIndexRange = True
    return isInsideIndexRange

# TODO:test this
def sort_uniq(sequence):
    return (x[0] for x in itertools.groupby(sorted(sequence)))


# TODO: implement this
def adjacentNodeToPotentialBombBlast(listOfBombs, potentialPath, player1indexes):

    # MoveToTheTileNextToMe(player1indexes,node)
    # rightward (0,1)
    # left (0,-1)
    # downward (1,0)
    # upward (-1,0)

    listOfAdjacents = []

    blastinPositions = potentialPathWithinBlasts(listOfBombs,potentialPath)

    neighboors = [(0,1),(0,-1),(1,0),(-1,0)]

    tp = tilePositionGenerator()

    # create everything as one
    # AdajacentNodesArray = np.ones_like(potentialPath)
    AdajacentNodesArray = potentialPath

    # set to zero: bomb blast and blocked tile
    for checkingAdjacentTile in tp:
        x = int( checkingAdjacentTile[0] / 32 )
        y = int( checkingAdjacentTile[1] / 32 )

        # if (potentialPath[y, x] == 0):
        #     AdajacentNodesArray[y, x] = 0
        # ff
        # next to a bomb blast
        if(blastinPositions[y,x]==1):
            # AdajacentNodesArray[y, x] = 0

            # check every neighboors
            for neighboor in neighboors:
                nodeTested = np.subtract(player1indexes, neighboor)
                print("nodeTested:",nodeTested)
                inRange = isIndexesRange(nodeTested)
                if(inRange==True):
                    print("if(isIndexesRange==True):")
                    # former test
                    if(potentialPath[nodeTested[0],nodeTested[1]]==1):
                        listOfAdjacents.append(nodeTested)

                    # # test if there is no blast onto the tested node
                    # if(potentialPath[nodeTested[0]],nodeTested[1]==1):
                    #     if(AdajacentNodesArray[nodeTested[0],nodeTested[1]]==1):
                    #         listOfAdjacents.append(nodeTested)
                else:
                    print("!if(isIndexesRange==True):")

    # sorting the points
    listOfAdjacents = sorted(listOfAdjacents , key=lambda k: [k[1], k[0]])

    return listOfAdjacents


# DONE: checked: ok
def potentialPathWithinBlasts(listOfBombs,potentialPath):
    pathInBlasts = np.zeros_like(potentialPath)
    for bombPosition in listOfBombs:
        xBomb = bombPosition[0]
        yBomb = bombPosition[1]
        # notsorted
        #
        # upwward
        # downward
        # rightward
        # leftward
        for i in range(4):
            xTmp = xBomb
            yTmp = yBomb
            while ((potentialPath[xTmp, yTmp] == 1) & (isIndexesRange((xTmp, yTmp)))):
                pathInBlasts[xTmp, yTmp] = 1
                if(i==0):
                    xTmp += 1
                if(i==1):
                    xTmp -= 1
                if(i==2):
                    yTmp += 1
                if(i==3):
                    yTmp -= 1


    return pathInBlasts
    # return []


previousBombPlacedPosition =(0,0)
currentBombPlacedPosition =(0,0)

# TODO: implement this
def closestNodeOutOfDanger(player1indexes,potentialPath):
    # player1indexes is onto bomb position
    print("closestNodeOutOfDanger")

    notPotentialPath = np.ones_like(potentialPath)
    np.place(notPotentialPath,potentialPath>0,0)

    # nextSteps = astar(notPotentialPath, (player1indexes[1], player1indexes[0]),
    #                   (previousBombPlacedPosition[0], previousBombPlacedPosition[1]))

    nextSteps = astar(notPotentialPath, (player1indexes[1], player1indexes[0]),
                      (previousBombPlacedPosition[0], previousBombPlacedPosition[1]))

    if(nextSteps!=False):
        print("nextSStep:",nextSteps[len(nextSteps)-1])

    # notPotentialPath[player1indexes[0],player1indexes[1]] = 3
    # notPotentialPath[:,player1indexes[1]] = 2
    # notPotentialPath[player1indexes[0],:] = 2

    # print("notPotentialPath:\n",notPotentialPath)
    # TODO: closest nodes


    # nextSteps = astar(notPotentialPath,(player1indexes[1],player1indexes[0]),(closestNodeToEnemy[0],closestNodeToEnemy[1]))

    return (0,0)
    pass

# TODO:implement this
def putBombAndStartToRunAway(node,player1indexes,potentialPath):
    print("putBombAndStartToRunAway")
    # keyboard.press_and_release('ctrl')
    keyboard.press('ctrl')
    time.sleep(0.15)
    keyboard.release('ctrl')

    currentBombPlacedPosition = player1indexes

    # MoveToTheTileNextToMe(player1indexes,node)
    # rightward (0,1)
    # left (0,-1)
    # downward (1,0)
    # upward (-1,0)
    neighboors = [(0,1),(0,-1),(1,0),(-1,0)]
    for i in neighboors:
        ii = np.subtract(player1indexes, i)
        print("ii:",ii)
        if(ii[0]>=0):
            if(ii[0]<=19):
                if(ii[1]>=0):
                    if(ii[1]<=15):
                        # print("ii[0] in [0-19]")
                        # print("ii[1] in [0-15]")
                        print("ii in (0,0) and (19,15)")

    # TODO: detect bomb position
    closestNodeOutOfDanger(player1indexes,potentialPath)


def GoToPositionOneStep(player1indexes,closestNodeToEnemy,potentialPath):

    # potentialPath.shape Out[2]: (15, 20)
    notPotentialPath = np.ones_like(potentialPath)
    np.place(notPotentialPath,potentialPath>0,0)
    # print(                       (player1indexes[1],player1indexes[0]),(closestNodeToEnemy[0],closestNodeToEnemy[1]))
    nextSteps = astar(notPotentialPath,(player1indexes[1],player1indexes[0]),(closestNodeToEnemy[0],closestNodeToEnemy[1]))
    print("nextSteps:",nextSteps)

    # TODO: implement
    if(nextSteps!=False):
        if(len(nextSteps)!=0):
            nextStep = nextSteps[len(nextSteps)-1]
            print("nextStep:",nextStep)
            MoveToTheTileNextToMe((player1indexes[1],player1indexes[0]),(nextStep[0],nextStep[1]))
        else:
            pass

    print("player1indexes,closestNodeToEnemy:",player1indexes,closestNodeToEnemy)
    if(player1indexes[1]==closestNodeToEnemy[0]):
        print("111")
        if(player1indexes[0]==closestNodeToEnemy[1]):
            print("222")
            putBombAndStartToRunAway((closestNodeToEnemy[0], closestNodeToEnemy[1]),
                                 (player1indexes[1], player1indexes[0]),potentialPath)
    else:
        if(nextSteps==False):
            print("333")
            putBombAndStartToRunAway((closestNodeToEnemy[0], closestNodeToEnemy[1]),
                                 (player1indexes[1], player1indexes[0]),potentialPath)



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

    # time.sleep(0.05)
    # keyboard.release('s')
    # keyboard.release('f')
    # keyboard.release('d')
    # keyboard.release('f')
    pass



def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

# DONE:implement this
# do one step toward to bomb something
def oneStepToPutBomb(potentialPath,potentialPathList,player1indexes,player2indexes,listOfBombs):
    # print()
    # potentialPath.shape Out[2]: (15, 20)
    print("closest_node:",closest_node(player2indexes,potentialPathList))
    closest_node1=closest_node(player2indexes,potentialPathList)

    # TODO: detected if a bomb is aligned with the controlled player
    aligned_with_bomb_blast = False
    for bombsPosition in listOfBombs:
        print("bombsPosition:",bombsPosition)
        if(player1indexes[1]==bombsPosition[0]):
            aligned_with_bomb_blast = True
        if (player1indexes[0] == bombsPosition[1]):
            aligned_with_bomb_blast = True
    if(aligned_with_bomb_blast==False):
        print("aligned_with_bomb_blast==False")
        GoToPositionOneStep(player1indexes,closest_node1,potentialPath)
    else:
        print("aligned_with_bomb_blast==True")
        nearestRunAwayNodes = adjacentNodeToPotentialBombBlast(listOfBombs, potentialPath, player1indexes)
        print("nearestRunAwayNodes:",nearestRunAwayNodes)
        # if(nearestRunAwayNodes!=[]):
        # Run_AwayNode = closest_node(player1indexes,nearestRunAwayNodes)
        print()
        # GoToPositionOneStep(player1indexes,Run_AwayNode,potentialPath)

    pass

def convertToIndexesGetPlayerPosition(getPlayerPosition):
    playerXindex = int( ( getPlayerPosition[0] - getPlayerPosition[0] % 32 )/32 )
    playerYindex = int( ( getPlayerPosition[1] - getPlayerPosition[1] % 32 )/32 )
    return (playerXindex,playerYindex)


def availiablePathToControlledPlayer(availiablePath, getPlayerPosition):
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

    availiablePathRet = np.zeros((15,20))

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

    availiableSpots = np.ones((15,20))

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
    screenAveragedRet = np.zeros((15,20,3))
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
        for ii in range(15):
            # print(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)
            yield(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)




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

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

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

    listOfBombs = listBombsPositions(screenAveraged)

    print("listOfBombs:",listOfBombs)

    availiablePath = AvailiablePath(screen,screenAveraged, 1)

    # print(availiablePath)

    potentialPathList,potentialPath = availiablePathToControlledPlayer(availiablePath, getPlayerPosition)

    # DONE: remove the bottom line
    # print(potentialPath)

    player1indexes = convertToIndexesGetPlayerPosition(GetPlayerPosition(screen,1))
    player2indexes = convertToIndexesGetPlayerPosition(GetPlayerPosition(screen,2))

    print("player1indexes:",player1indexes)
    print("player2indexes:",player2indexes)


    oneStepToPutBomb(potentialPath,potentialPathList,player1indexes,player2indexes,listOfBombs)



    time.sleep(1)


    if (keyboard.is_pressed('p') == True):
        paused = True
        cv2.destroyAllWindows()
        break

    # too see what is captured
    # cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    # cv2.imshow('thresh', thresh)

    # if cv2.waitKey(25) & 0xFF == ord('t'):
    #     cv2.destroyAllWindows()
    #     break

