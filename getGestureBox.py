from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import sklearn.neighbors
import numpy as np
import pickle
import sys
import ast
import queue

gestureArea = []

def createBox (event, x,y, flags, params):
    global gestureArea
    if event == cv.EVENT_LBUTTONDOWN:
        gestureArea = [(x, y)]
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if (x < 0):
            x = 0
        if (y < 0):
            y = 0
        if (x > frame.shape[1]):
            x = frame.shape[1] - 1
        if (y > frame.shape[0]):
            y = frame.shape[0] - 1
        gestureArea.append((x, y))
        gFile = open("gestureBox.txt", 'w')
        gFile.write("%s\n" % str(gestureArea[0]))
        gFile.write("%s\n" % str(gestureArea[1]))
        gFile.close()
        # draw a rectangle around the region of interest
        cv.putText(frame, "Click and drag the area where you will perform gestures",(10,25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
        cv.rectangle(frame, gestureArea[0], gestureArea[1], (0, 255, 0), 1)
        cv.imshow("Frame", frame)

#creating camera object
capture = cv.VideoCapture(0)
frameQ = queue.Queue()

while True:
    ret, frame = capture.read()
    frameShow = frame.copy()
    if frame is None:
        break
    cv.putText(frameShow, "Click and drag the area where you will perform gestures",(10,25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
    cv.putText(frameShow, "Press P when you have the correct size and location ",(10,50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
    if (len(gestureArea)==2):
        cv.rectangle(frameShow, gestureArea[0], gestureArea[1], (0, 255, 0), 1)
    cv.imshow("Frame", frameShow)
    cv.setMouseCallback("Frame", createBox)
    keyboard = cv.waitKey(10)
    if keyboard == ord('p'):
        cv.imwrite("background.png", frame)
        break  
