import cv2       
import sys                      
import numpy as np #importing libraries
np.set_printoptions(threshold=sys.maxsize)

cap = cv2.VideoCapture(0) #creating camera object
while( cap.isOpened() ) :
    ret,gray = cap.read() #reading the frames
    #img = cv2.flip(img, 1) #flip image
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray
    #ret,gray = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #gray = 255-gray
    cv2.imshow('input',gray) #displaying the frames
    k = cv2.waitKey(10)
    if k == ord('q'): #exit when q is pressed
        cv2.imwrite("./testdata/test.png", gray)
        #f=open("./testdata/test.txt", "w")
        #f.write("[")
        #f.write(np.array2string(gray, separator=", "))
        #f.write("]")
        #f.close()
        break
