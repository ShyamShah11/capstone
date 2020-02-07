import cv2              
import sys               
import os
import numpy as np #importing libraries
import time
np.set_printoptions(threshold=sys.maxsize)
total_files = 10

def convert_to_data(directory_name, gesture, output):
    num_files = 0
    f=open(output, "w") #switch to "a" to append
    f.write("[")
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        if num_files >= total_files:
            break
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            img = cv2.imread(directory_name+filename) #reading the frames
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray
            f.write(np.array2string(gray, separator=", "))
            if num_files +1 < total_files:
                f.write(", \n")
        else:
            continue
        num_files += 1

    
    f.write("]")
    f.close()

if __name__ == "__main__":
    data = convert_to_data("./gestures/leapGestRecog/00/01_palm/", "palm", "./data/palm.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/02_l/", "l", "./data/l.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/03_fist/", "fist", "./data/fist.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/04_fist_moved/", "fist_moved", "./data/fist_moved.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/05_thumb/", "thumb", "./data/thumb.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/06_index/", "index", "./data/index.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/07_ok/", "ok", "./data/ok.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/08_palm_moved/", "palm_moved", "./data/palm_moved.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/09_c/", "c", "./data/c.txt")
    data = convert_to_data("./gestures/leapGestRecog/00/10_down/", "down", "./data/down.txt")
