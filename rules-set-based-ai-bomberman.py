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


previousPlayer1Position = (0,0)

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


# TODO: fixes thresholds?
def listBombsPositions(screenAvged):
    # print("listBombsPostions")
    tp = tilePositionGenerator()
    list = []
    for potentialBombs in tp:
        x = int( potentialBombs[0] / 32 )
        y = int( potentialBombs[1] / 32 )

        # [45.29032258 55.72528616 62.90114464]
        # [45.29032258 53.87721124 54.82622268]
        # [46.27263267 56.09157128 57.52341311]
        # [46.15608741 54.89281998 56.05827263]
        # [47.421436  57.3735692 59.2049948]
        # [48.4037461  58.98855359 60.85327784]
        # [48.08740895 58.87200832 60.35379813]
        # [49.06971904 60.13735692 61.8855359]
        #  74.52653485952133 176.08324661810613 204.88657648283038
        #  91.69198751300729 169.4901144640999 193.06555671175857
        #  107.4588969823101 163.92924037460978 181.86056191467222

        screenAveraged = screenAvged.astype(int)

        if (screenAveraged[y, x, 0] > 40):
            if(screenAveraged[y,x,0]<=55):
                if (screenAveraged[y, x, 1] > 48):
                    if(screenAveraged[y,x,1]<=66):
                        if(screenAveraged[y,x,2]>49):
                            if(screenAveraged[y,x,2]<=68):
                        # if (screenAveraged[y, x, 2] > 49):
                        #     if(screenAveraged[y,x,2]<=125):
                                # print("bomb decteted")
                                list.append([y,x])

        if (int(screenAveraged[y, x, 0]) == 74):
            if (int(screenAveraged[y, x, 0]) == 176):
                if (int(screenAveraged[y, x, 0]) == 204):
                    list.append([y,x])

        if (int(screenAveraged[y, x, 0]) == 91):
            if (int(screenAveraged[y, x, 0]) == 169):
                if (int(screenAveraged[y, x, 0]) == 193):
                    list.append([y, x])

        if (int(screenAveraged[y, x, 0]) == 107):
            if (int(screenAveraged[y, x, 0]) == 163):
                if (int(screenAveraged[y, x, 0]) == 181):
                    list.append([y, x])

    return list


def isIndexesRange(point):
    isInsideIndexRange = False
    if (point[1] >= 0):
        if (point[1] < 20):
            if (point[0] >= 0):
                if (point[0] < 15):
                    # print("ii in (0,0) and (19,15)")
                    isInsideIndexRange = True
    return isInsideIndexRange

# DONE:test this
def sort_uniq(sequence):
    return (x[0] for x in itertools.groupby(sorted(sequence)))


# DONE: implement this
def adjacentNodeToPotentialBombBlast(listOfBombs, potentialPath, player1indexes):

    listOfAdjacents = []

    blastinPositions = potentialPathWithinBlasts(listOfBombs,potentialPath)

    print("blastinPositions:\n",blastinPositions)

    neighboors = [(0,1),(0,-1),(1,0),(-1,0)]

    tp = tilePositionGenerator()

    # set to zero: bomb blast and blocked tile
    for checkingAdjacentTile in tp:
        x = int( checkingAdjacentTile[0] / 32 )
        y = int( checkingAdjacentTile[1] / 32 )

        # next to a bomb blast
        if(blastinPositions[y,x]==1):
            print("====================")
            print("[y,x]     :",[y,x])

            # check every neighboors
            for neighboor in neighboors:
                # print("neighboor:",neighboor)
                # nodeTested = np.subtract([x,y], neighboor)
                nodeTested = np.subtract([y,x], neighboor)
                print("nodeTested:",nodeTested)
                inRange = isIndexesRange(nodeTested)
                if(inRange==True):
                    # 7th iter
                    if(potentialPath[nodeTested[0],nodeTested[1]]==1):
                        if(blastinPositions[nodeTested[0],nodeTested[1]]==0):
                            listOfAdjacents.append(nodeTested)
                else:
                    # print("!if(isIndexesRange==True):")
                    pass

    # sorting the points: not really usefull anymore
    listOfAdjacents = sorted(listOfAdjacents , key=lambda k: [k[1], k[0]])

    return listOfAdjacents


# DONE: checked: ok
def potentialPathWithinBlasts(listOfBombs,potentialPath):
    pathInBlasts = np.zeros_like(potentialPath)
    for bombPosition in listOfBombs:
        yBomb = bombPosition[0]
        xBomb = bombPosition[1]

        # notsorted
        # TODO: sort the result
        # upwward, downward, rightward, leftward
        for i in range(4):
            xTmp = xBomb
            yTmp = yBomb
            # DONE bugfix: while ((potentialPath[xTmp, yTmp] == 1) & (isIndexesRange((xTmp, yTmp)))):
            # DONE bugfix: IndexError: index 15 is out of bounds for axis 0 with size 15
            while ((potentialPath[yTmp, xTmp] == 1) and (isIndexesRange((yTmp, xTmp))==True)):
                pathInBlasts[yTmp, xTmp] = 1
                if (i == 0):
                    xTmp += 1
                    if(isIndexesRange((0,xTmp))==False):
                        break
                if (i == 1):
                    xTmp -= 1
                    if(isIndexesRange((0,xTmp))==False):
                        break
                if (i == 2):
                    yTmp += 1
                    if(isIndexesRange((yTmp,0))==False):
                        break
                if (i == 3):
                    yTmp -= 1
                    if(isIndexesRange((yTmp,0))==False):
                        break
                # print("[yTmp, xTmp]:",[yTmp, xTmp])

    return pathInBlasts
    # return []


previousBombPlacedPosition =(0,0)
currentBombPlacedPosition =(0,0)
pathLength = 0

def putBombAndStartToRunAway(player1indexes,node,potentialPath):
    print("putBombAndStartToRunAway")
    keyboard.press('ctrl')
    time.sleep(0.15)
    keyboard.release('ctrl')

    currentBombPlacedPosition = player1indexes

    MoveToTheTileNextToMe(player1indexes, node)


def GoToPositionOneStep(player1indexes,closestNodeToEnemy,potentialPath):

    # potentialPath.shape Out[2]: (15, 20)
    notPotentialPath = np.ones_like(potentialPath)
    np.place(notPotentialPath,potentialPath>0,0)

    nextSteps = astar(notPotentialPath,(player1indexes[0],player1indexes[1]),(closestNodeToEnemy[0],closestNodeToEnemy[1]))
    print("nextSteps:",nextSteps)

    global previousPlayer1Position
    global pathLength
    # TODO: implement
    if(nextSteps!=False):
        if(len(nextSteps)!=0):
            nextStep = nextSteps[len(nextSteps)-1]
            print("nextStep:",nextStep)
            MoveToTheTileNextToMe(player1indexes,nextStep)
            previousPlayer1Position = player1indexes
            pathLength = len(nextSteps)
        else:
            pass


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



def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


previousPlayer1Position = (0,0)

# DONE:implement this
# do one step toward to bomb something
def oneStepToPutBomb(potentialPath,potentialPathList,
                     player1indexes,player2indexes,listOfBombs,getPlayerPosition):
    # potentialPath.shape Out[2]: (15, 20)
    # objective to bomb
    closest_node1=closest_node(player2indexes,potentialPathList)
    # print("closest_node1:",closest_node1)

    global previousPlayer1Position

    # blast array ?
    blastinPositions = potentialPathWithinBlasts(listOfBombs, potentialPath)

    if(blastinPositions[closest_node1[0],closest_node1[1]]==0):
        # print("if(blastinPositions[closest_node1[0],closest_node1[1]]==0):")
        # print(closest_node1,player1indexes)
        if(np.array_equal(closest_node1,player1indexes)):
            # print("if(closest_node1==player1indexes):")
            # putBomb
            putBombAndStartToRunAway(player1indexes,closest_node1,potentialPath)
            # runAway
            GoToPositionOneStep(player1indexes,previousPlayer1Position,potentialPath)
        else:
            # print("!if(closest_node1==player1indexes):")
            # goToTile
            GoToPositionOneStep(player1indexes,closest_node1,potentialPath)

    if(blastinPositions[player1indexes[0],player1indexes[1]]==1):
        # print("if(blastinPositions[player1indexes[0],player1indexes[1]]):")
        # go at a adjacent position
        # runAway
        nearestRunAwayNodes = adjacentNodeToPotentialBombBlast(listOfBombs, potentialPath, player1indexes)
        # print("nearestRunAwayNodes:",nearestRunAwayNodes)
        Run_AwayNode = closest_node(player1indexes,nearestRunAwayNodes)
        GoToPositionOneStep(player1indexes,Run_AwayNode,potentialPath)

    # print("getPlayerPosition:", getPlayerPosition)
    tmpCoincoin = np.subtract(getPlayerPosition, [player1indexes[0] * 32, player1indexes[1] * 32])
    # print("tmpCoincoin:", tmpCoincoin)

    timeToUnstuck = 0.05
    # if(tmpCoincoin[0]<5):
    if(tmpCoincoin[0]<6):
        # print("stuck?")
        keyboard.press('d')
        time.sleep(timeToUnstuck)
        keyboard.release('d')
    # if(tmpCoincoin[0]>27):
    if(tmpCoincoin[0]>26):
        # print("stuck?")
        keyboard.press('e')
        time.sleep(timeToUnstuck)
        keyboard.release('e')
    # if(tmpCoincoin[1]<5):
    if(tmpCoincoin[0]<6):
        # print("stuck?")
        keyboard.press('f')
        time.sleep(timeToUnstuck)
        keyboard.release('f')
    # if(tmpCoincoin[1]>27):
    if(tmpCoincoin[0]>26):
        # print("stuck?")
        keyboard.press('s')
        time.sleep(timeToUnstuck)
        keyboard.release('s')

    previousPlayer1Position = player1indexes

    pass

def convertToIndexesGetPlayerPosition(getPlayerPosition):
    playerXindex = int( ( getPlayerPosition[0] - getPlayerPosition[0] % 32 )/32 )
    playerYindex = int( ( getPlayerPosition[1] - getPlayerPosition[1] % 32 )/32 )
    return (playerXindex,playerYindex)


def availiablePathToControlledPlayer(availiablePath, getPlayerPosition):
    playerIndexPos = convertToIndexesGetPlayerPosition(getPlayerPosition)
    playerYindex = playerIndexPos[0]
    playerXindex = playerIndexPos[1]

    # print(playerYindex,playerXindex)
    # print("getPlayerPosition:",getPlayerPosition)

    # print("availiablePath.shape:",availiablePath.shape)

    labeled = measure.label(availiablePath, background=False, connectivity=1)
    # reversed X,Y why ?

    # print("labeled.shape:",labeled.shape)
    # on the bottom line
    label = labeled[playerYindex, playerXindex]  # known pixel location

    rp = measure.regionprops(labeled)
    props = rp[label - 1]  # background is labeled 0, not in rp

    # props.bbox  # (min_row, min_col, max_row, max_col)
    # props.image  # array matching the bbox sub-image
    # print(len(props.coords))  # list of (row,col) pixel indices
    regionSize = len(props.coords)

    availiablePathRet = np.zeros((15,20))

    connectedCoords = props.coords
    for coord in connectedCoords:
        availiablePathRet[coord[0],coord[1]] = 1

    return regionSize,connectedCoords,availiablePathRet

# support player 1, 2
def GetPlayerPosition(screen, number):
    # number 1 2 3 4
    #DONE: fix position detection when the red player is underneath (same tile)
    # the green player
    # TODO: see to remember the last known position or ajust level to get more
    # fiting threshold

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

    if(len(cnts)==0):
        print("if(len(cnts)==0):\nred player is hidden underneath green ?\n")
        # returning green player position
        return GetPlayerPosition(screen, 2)


    # y,x convention

    # if player1 and 2 cross each other
    # bigfixe for out of bound output...
    if(sum_y>480-1):
        sum_y = 480-1
    if(sum_x>640-1):
        sum_x = 640-1

    return (sum_y,sum_x)


# DONE: fix the "filled" (even though clear) bottom path issue
# TODO: make it so it could process any player
def AvailiablePath(screen,screenAveraged,number,listOfBombs):
    crate = [65,151,191]
    hardBlock = [156,156,156]

    bomb01 = [45,55,62]
    bomb02 = [45,53,54]
    bomb03 = [46,56,57]
    bomb04 = [46,54,56]

    bomb   = [46,56,58]

    bomb05 = [47,57,59]
    bomb06 = [48,58,60]
    bomb07 = [48,58,60]
    bomb08 = [49,60,61]
    bomb09 = [74,176,204]
    bomb10 = [91,169,193]
    bomb11 = [107,163,181]

    # crate01 = [46, 152, 211]
    # crate02 = [65, 151, 191]
    # crate03 = [12, 188, 243]
    # crate04 = [0,  220, 252]

    crate01 = [46, 152, 211]
    crate02 = [65, 151, 191]
    crate03 = [12, 188, 243]
    crate04 = [0,  220, 252]
    crate05 = [32, 154, 225]

    crate06 = [71, 174, 202]
    crate07 = [74, 176, 205]
    crate08 = [88, 167, 191]
    crate09 = [93, 170, 193]

    # from the top line
    blast01_01 = [104, 162, 180]
    blast01_02 = [109, 165, 183]
    blast01_03 = [107, 163, 181]

    blast02_01 = [ 91, 169, 193]
    blast02_02 = [ 89, 168, 191]
    blast02_03 = [ 93, 170, 193]
    blast02_04 = [ 90, 168, 192]

    blast03_01 = [ 37, 200, 228]
    blast03_02 = [ 36, 199, 227]
    blast03_03 = [ 36, 199, 228]

    # from a vertical line
    blast04_01 = [111, 165, 183]
    blast04_02 = [104, 162, 180]

    blast05_01 = [ 93, 170, 194]
    blast05_02 = [ 88, 167, 191]

    blast06_01 = [ 37, 200, 228]
    blast06_02 = [ 35, 199, 227]

    # [45.29032258 55.72528616 62.90114464]
    # [45.29032258 53.87721124 54.82622268]
    # [46.27263267 56.09157128 57.52341311]
    # [46.15608741 54.89281998 56.05827263]
    # [47.421436  57.3735692 59.2049948]
    # [48.4037461  58.98855359 60.85327784]
    # [48.08740895 58.87200832 60.35379813]
    # [49.06971904 60.13735692 61.8855359]
    #  74.52653485952133 176.08324661810613 204.88657648283038
    #  91.69198751300729 169.4901144640999 193.06555671175857
    #  107.4588969823101 163.92924037460978 181.86056191467222

    screenAveragedToInt = screenAveraged.astype(int)

    # allBlocking = [crate,hardBlock]
    # allBlocking = [crate,hardBlock,bomb]
    # allBlocking = [crate,hardBlock,bomb,bomb01,bomb02,bomb03,bomb04,bomb05,
    #                bomb06,bomb07,bomb08,bomb09,bomb10,bomb11,
    #                crate01,crate02,crate03,crate04]
    # allBlocking = [crate,hardBlock,bomb,bomb01,bomb02,bomb03,bomb04,bomb05,
    #                bomb06,bomb07,bomb08,bomb09,bomb10,bomb11,
    #                crate01,crate02,crate03,crate04,crate05,crate06,crate07,crate08,crate09]
    allBlocking = [crate, hardBlock,
                    bomb, bomb01, bomb02, bomb03, bomb04, bomb05,
                    bomb06,bomb07,bomb08,bomb09,bomb10,bomb11,
                    blast01_01,blast01_02,blast01_03,
                    blast02_01,blast02_02,blast02_03,blast02_04,
                    blast03_01,blast03_02,blast03_03
                    ,blast04_01,blast04_02
                    ,blast05_01,blast05_02
                    ,blast06_01,blast06_02]


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
                break
            # else:
            #     print("!True\n")
        # if(IsItABomb(screenAveragedToInt[y,x])==True):
        #     availiableSpots[y,x] = False
    # for bomb in listOfBombs:
    #     print("bomb:",bomb)
    #     y = bomb[0]
    #     x = bomb[1]
    #     availiableSpots[y,x] = False

    return availiableSpots

def ScreenAveraging(screen):

    # print("screen.shape:",screen.shape)

    # tile of 32 piwel
    tileWidth = 32
    screenAveragedRet = np.zeros((15,20,3))
    tilePositons = tilePositionGenerator()

    # print("np.average:",np.average(screen[0:31,0:31,0]),np.average(screen[0:31,0:31,1]),np.average(screen[0:31,0:31,2]))

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
    for ii in range(15):
        for i in range(20):
            # print(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)
            yield(i*tileWidth,ii*tileWidth,(i+1)*tileWidth-1,(ii+1)*tileWidth-1)


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

stop = False

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

# TODO: implement this
def IsItABomb(pixel):
    # [45.29032258 55.72528616 62.90114464]
    # [45.29032258 53.87721124 54.82622268]
    # [46.27263267 56.09157128 57.52341311]
    # [46.15608741 54.89281998 56.05827263]
    # [47.421436  57.3735692 59.2049948]
    # [48.4037461  58.98855359 60.85327784]
    # [48.08740895 58.87200832 60.35379813]
    # [49.06971904 60.13735692 61.8855359]
    #  74.52653485952133 176.08324661810613 204.88657648283038
    #  91.69198751300729 169.4901144640999 193.06555671175857
    #  107.4588969823101 163.92924037460978 181.86056191467222

    if (pixel[0] > 40):
        if (pixel[0] <= 55):
            if (pixel[1] > 48):
                if (pixel[1] <= 66):
                    # if (pixel[2] > 49):
                    #     if (pixel[2] <= 68):
                    if (pixel[2] > 49):
                        if(pixel[2]<=125):
                            return True

    if (pixel[0] == 74):
        if(pixel[1] == 176):
            if(pixel[2] == 204):
                print("here 74.52653485952133 176.08324661810613 204.88657648283038")
                return True

    if(pixel[0] == 91):
        if(pixel[1] == 169):
            if(pixel[2] == 193):
                print("here 91.69198751300729 169.4901144640999 193.06555671175857")
                return True

    if(pixel[0] == 107):
        if(pixel[1] == 163):
            if(pixel[2] == 181):
                print("here 107.4588969823101 163.92924037460978 181.86056191467222")
                return True
    return False

# camera = cv2.VideoCapture("Bomber 2018-07-05 21-35-23-13.avi")

while True:

    # getting the window mode screen
    screen = grab_screen(region=(anchorHWidthTopLeft, anchorHeightTopLeft,
                                 anchorWidthBotRight, anchorHeightBotRight))

    # pixels characters 2
    # screen = cv2.resize(screen, (int(WIDTH / 2), int(HEIGTH / 2)))
    # screen = cv2.resize(screen, (int(WIDTH / 8), int(HEIGTH / 8)))

    # run a color convert:
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    # ret, screen = camera.read()

    getPlayerPosition = GetPlayerPosition(screen, 1)

    print()
    print("getPlayerPosition:",getPlayerPosition)

    screenAveraged = ScreenAveraging(screen)

    # print("screenAveraged:",screenAveraged)f

    listOfBombs = listBombsPositions(screenAveraged)

    # print("listOfBombs:",listOfBombs)

    # availiablePath = AvailiablePath(screen,screenAveraged, 1)
    availiablePath = AvailiablePath(screen,screenAveraged, 1,listOfBombs)

    # print("availiablePath:\n",availiablePath)

    # print("screenAveraged[0,:].astype(int):\n",screenAveraged[0,:].astype(int))
    # print("screenAveraged[:,8].astype(int):\n",screenAveraged[:,8].astype(int))

    regionSize,potentialPathList,potentialPath = availiablePathToControlledPlayer(availiablePath, getPlayerPosition)

    # DONE: remove the bottom line
    # print("regionSize:",regionSize)
    # print("potentialPath:\n",potentialPath)

    player1indexes = convertToIndexesGetPlayerPosition(GetPlayerPosition(screen,1))
    player2indexes = convertToIndexesGetPlayerPosition(GetPlayerPosition(screen,2))

    # print("player1indexes:",player1indexes)
    # print("player2indexes:",player2indexes)


    oneStepToPutBomb(potentialPath,potentialPathList,player1indexes,player2indexes,listOfBombs,getPlayerPosition)

    # time.sleep(1)

    if (keyboard.is_pressed('p') == True):
        paused = True
        cv2.destroyAllWindows()
        break

    # too see what is capturedd
    # cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    # cv2.imshow('thresh', thresh)

    # if cv2.waitKey(25) & 0xFF == ord('t'):
    #     cv2.destroyAllWindows()
    #     break

