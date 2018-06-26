
import keyboard

# default player keyboard binding
e =     [1,0,0,0,0,0]
d =     [0,1,0,0,0,0]
s =     [0,0,1,0,0,0]
f =     [0,0,0,1,0,0]
ctrl =  [0,0,0,0,1,0]
shift = [0,0,0,0,0,1]

all_keys = [e,d,s,f,ctrl,shift]

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
    if(keyboard.is_pressed('shift')):
        return shift
    pass

# def press_a_key(key):
#     if(key.):
#         keyboard.press_and_release()