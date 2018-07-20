import ctypes as c
from ctypes import wintypes as w
from struct import *
from time import *
import datetime
import sys

import keyboard
import time

# from https://codereview.stackexchange.com/questions/120528/implementing-a-memory-scanner-in-python

pid = 1960

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
log=open(r'out.txt.','wb',0)

last_time = time.time()

for i in addresses_list:
    ReadProcessMemory(ph, c.c_void_p(i), buff, bufferSize, c.byref(bytesRead))
    value = unpack('I',buff)[0]
    # print("lol")
    # print("i",i)
    if(0 == value):
        print(value, "i in hex",'%s' % hex(i))

    if(608 == value):
        # print(value)
        # print("i",i)
        print(value, "i in hex",'%s' % hex(i))
        listOf680.append(i)
    if(i % int(0x1000000) == 0):
        print("%",i)
    if(i > 0x200000):
        break
        # break
    # if value == int(sys.argv[1]):
    #     log.write('%x\r\n' % (i, ))

# i is 0019F7D0
# myst 000440B0
# offset = i - int(0x0019F7D0 - 0x000440B0)

print(time.time()-last_time)


stop = False
while(stop ==False):
    # tmpOffset = i - int(0x0019F7A0) + int(0x440B0)
    # tmpOffset = int(0x440B0)
    # tmpOffset = int(0x55C9C)
    tmpOffset = int(0x4440B0)
    print("tmpOffset",'%s' % hex(tmpOffset))

    ReadProcessMemory(ph, c.c_void_p(tmpOffset), buff, bufferSize, c.byref(bytesRead))
    value = unpack('I',buff)[0]
    print("value",value)

    sleep(0.2)

    if(keyboard.is_pressed('n')):
        stop=True
    pass
