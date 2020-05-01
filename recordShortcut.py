import sys
import threading
import mouse
import keyboard
import time
from pickle import dump

def record_Shortcut(name):
    #create empty lists for the sequences of mouse and keyboard inputs
    mouse_events = []
    keyboard_events = []
    #waits for special keypress before starting the recording
    print("Press the '*' key when you would like to start recording")
    keyboard.wait('*')
    mouse.hook(mouse_events.append)

    #waits until the same keypress to stop recording
    keyboard_events = keyboard.record(until='*')
    mouse.unhook(mouse_events.append)

    print("Recording done!")

    #print statements for testing you can remove these
    #-----------------------------------------------
    print(str(mouse_events))
    print(str(keyboard_events))
    #-----------------------------------------------
    
    #takes sequence of key and mouse presses as well as the shortcut name and stores in respective files
    nfile = open("name_shortcuts.txt",'a+')
    nfile.write("%s\n" % (name))
    nfile.close()

    mfile = open("mouse_shortcuts.txt",'ab+')
    dump(mouse_events,mfile)
    mfile.close()

    kfile = open("keyboard_shortcuts.txt",'ab+')
    dump(keyboard_events,kfile)
    kfile.close()

record_Shortcut(sys.argv[1])
