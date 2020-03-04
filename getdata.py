import cv2              
import sys               
import os
import numpy as np #importing libraries
import time
np.set_printoptions(threshold=sys.maxsize)
total_files = 200

def convert_to_data(directory_name, gesture, output):
    f=open(output, "wb") #switch to "a" to append
    thresh = None
    for i in range(10):
        num_files = 0
        directory = os.fsencode(directory_name+str(i)+gesture)
        for file in os.listdir(directory):
            if num_files >= total_files:
                break
            filename = os.fsdecode(file)
            if filename.endswith(".png"): 
                img = cv2.imread(directory_name+filename) #reading the frames
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray
                if (thresh is None):
                    ret,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
               else:
                    ret, threshNew = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                    if(thresh.shape == threshNew.shape):
                        thresh = np.stack((thresh, threshNew), axis=0)
                    else:
                        thresh = np.concatenate((thresh, [threshNew]))
            else:
                continue
            num_files += 1

    np.save(f, thresh)
    f.close()

def create_white (output):
    f=open(output, "wb") #switch to "a" to append
    thresh = np.zeros((2000,240,640))
    np.save(f, thresh)
    f.close()
    
if __name__ == "__main__":
    data = convert_to_data("./gestures/leapGestRecog/0", "/01_palm/", "./data/palm")
    data = convert_to_data("./gestures/leapGestRecog/0", "/02_l/", "./data/l")
    data = convert_to_data("./gestures/leapGestRecog/0", "/03_fist/", "./data/fist")
    data = convert_to_data("./gestures/leapGestRecog/0", "/05_thumb/", "./data/thumb")
    data = convert_to_data("./gestures/leapGestRecog/0", "/06_index/", "./data/index")
    data = convert_to_data("./gestures/leapGestRecog/0", "/07_ok/", "./data/ok")
    data = convert_to_data("./gestures/leapGestRecog/0", "/09_c/", "./data/c")
    data = convert_to_data("./gestures/leapGestRecog/0", "/10_down/", "./data/down")
    create_white('./data/white')
