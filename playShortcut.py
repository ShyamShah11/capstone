import threading
import mouse
import keyboard
import time

def play_shortcut(mouse_events,keyboard_events):

    #Uses threading to play both mouse and keyboard movements at the same time
    m_thread = threading.Thread(target = lambda :mouse.play(mouse_events,speed_factor=2))
    k_thread = threading.Thread(target = lambda :keyboard.play(keyboard_events,speed_factor=2))
    
    m_thread.start()
    k_thread.start()

    m_thread.join()
    k_thread.join() 
