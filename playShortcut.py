import threading
import mouse
import keyboard
import time
from pickle import load
def play_shortcut(name):
    selectedShortcut = 0
    totalShortcuts = 0
    with open("name_shortcuts.txt",'r') as nfile:
        for line in nfile.readlines():
            if name in line:
                selectedShortcut = totalShortcuts
            totalShortcuts += 1
    
    kfile = open("keyboard_shortcuts.txt",'rb+')
    for i in range (selectedShortcut + 1):
        keyboard_events = load(kfile)
    kfile.close()

    mfile = open("mouse_shortcuts.txt",'rb+')
    for j in range (selectedShortcut + 1):
        mouse_events = load(mfile)
    mfile.close()

    #just to see what inputs will be done can be removed
    #--------------------------------------------------
    print(str(mouse_events))
    print(str(keyboard_events))
    #--------------------------------------------------

    #Uses threading to play both mouse and keyboard movements at the same time
    m_thread = threading.Thread(target = lambda :mouse.play(mouse_events,speed_factor=1))
    k_thread = threading.Thread(target = lambda :keyboard.play(keyboard_events,speed_factor=1))
    m_thread.start()
    #This one line of code fixed the keyboard inputs not working (even though the mouse inputs did)
    keyboard._listener.start_if_necessary()
    k_thread.start()
    m_thread.join()
    k_thread.join() 
#was used for testing
#play_shortcut("Grindin'")
