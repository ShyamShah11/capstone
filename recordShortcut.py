import threading
import mouse
import keyboard
import time
from playShortcut import play_shortcut

#create empty lists for the sequences of mouse and keyboard inputs
mouse_events = []
keyboard_events = []

#Asks for a name to associate to the shortcut
name = input("Enter Shortcut Name : ")
#waits for special keypress before starting the recording
keyboard.wait("page up")
mouse.hook(mouse_events.append)

#waits until the same keypress to stop recording
keyboard_events = keyboard.record(until="page up")
mouse.unhook(mouse_events.append)


#takes sequence of key and mouse presses as well as the shortcut name and stores in respective text files
nfile = open("name_shortcuts.txt","a+")
nfile.write("%s\n" % (name))
nfile.close()

mfile = open("mouse_shortcuts.txt","a+")
mfile.write("%s\n" % str(mouse_events))
mfile.close()

kfile = open("keyboard_shortcuts.txt","a+")
kfile.write("%s\n" % str(keyboard_events))
kfile.close()

#plays what was just recorded (for the purpose of testing)
play_shortcut(mouse_events,keyboard_events)