import cv2                             
import numpy as np #importing libraries
cap = cv2.VideoCapture(0) #creating camera object
while( cap.isOpened() ) :
    ret,img = cap.read() #reading the frames
    img = cv2.flip(img, 1) #flip image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray
    ret,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('input',thresh) #displaying the frames
    k = cv2.waitKey(10)
    if k == ord('q'): #exit when q is pressed
        break
