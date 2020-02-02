import cv2              
import sys               
import os
import numpy as np #importing libraries
np.set_printoptions(threshold=sys.maxsize)
total_files = 10

def convert_to_data(directory_name, gesture, output):
    num_files = 0
    f=open(output, "a")
    f.write("[")
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        if num_files >= total_files:
            break
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            img = cv2.imread(directory_name+filename) #reading the frames
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray
            ret,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            thresh = ~thresh #invert colors
            #cv2.imshow('input',thresh) #displaying the frames
            f.write(np.array2string(thresh, separator=", "))
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
