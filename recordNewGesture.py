from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import sklearn.neighbors
import numpy as np
import pickle
import sys
import ast
import queue

#Obtaining the gesture area from settings file
gFile = open("gestureBox.txt", 'r')
gestureAreaX = ast.literal_eval((gFile.readline()))
gestureAreaY = ast.literal_eval((gFile.readline()))
gestureArea = [gestureAreaX, gestureAreaY]

#Obtaining the background image
back = cv.imread("background.png")
background =  back[gestureArea[0][1]:gestureArea[1][1], gestureArea[0][0]:gestureArea[1][0]]
background = cv.GaussianBlur(background,(21,21), 0)

#creating camera object
capture = cv.VideoCapture(0)
frameQ = queue.Queue()
frameCount = 0

def recordGestures(fileName):
    print('here')
    frameCount = 0
    timeCount = 1
    while True:
        if frameCount < 360:
            ret, frame = capture.read()
            if frame is None:
                break
            print(frameCount)
            print(frameCount % 60)

            cv.rectangle(frame, gestureArea[0], gestureArea[1], (0, 255, 0), 1) #adding the gesture area to camera
            roiFrame = frame[gestureArea[0][1]:gestureArea[1][1], gestureArea[0][0]:gestureArea[1][0]] #gesture area of current frame
            fgMask = cv.GaussianBlur(roiFrame,(21,21), 0) #blurring it to reduce sensitivity to noise

            delta = cv.absdiff(fgMask, background) #getting the difference between the current frame and the background frame
            ret,thresh = cv.threshold(delta, 20, 255, cv.THRESH_BINARY) #creating harsh contrast for the difference
            thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY) #converting to greyscale
            thresh = cv.resize(thresh, (150,150)) #resizing the image for processing

            #show the current recording
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', thresh)
            cv.imshow('delta', delta)
            frameCount +=1
            #fileName = np.concatenate((fileName, [thresh]))

saveImages = False
counter = 1
frameCount = 0
fileName = None
timeCount = 1
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    cv.rectangle(frame, gestureArea[0], gestureArea[1], (0, 255, 0), 1) #adding the gesture area to camera
    cv.imshow('Frame', frame)
    keyboard = cv.waitKey(10)
    if keyboard == ord('p') and counter <=5:
        saveImages = True
    if saveImages:
        #cv.putText(frame, str(timeCount),(10,50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
        cv.imshow('Frame', frame)
        #print(frameCount)
        if frameCount < 360:
            if (frameCount % 60 == 0):
                timeCount +=1
            roiFrame = frame[gestureArea[0][1]:gestureArea[1][1], gestureArea[0][0]:gestureArea[1][0]] #gesture area of current frame
            fgMask = cv.GaussianBlur(roiFrame,(21,21), 0) #blurring it to reduce sensitivity to noise

            delta = cv.absdiff(fgMask, background) #getting the difference between the current frame and the background frame
            ret,thresh = cv.threshold(delta, 20, 255, cv.THRESH_BINARY) #creating harsh contrast for the difference
            thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY) #converting to greyscale
            thresh = cv.resize(thresh, (150,150)) #resizing the image for processing

            #show the current recording
            cv.imshow('FG Mask', thresh)
            cv.imshow('delta', delta)
            frameCount +=1
            if fileName is None:
                fileName = [thresh]
            else:
                #print("check")
                fileName = np.concatenate((fileName, [thresh]))
        else:
            saveImages = False
            frameCount = 0
            counter += 1
            timeCount = 1 
    if (counter>=5):
        f = open(sys.argv[1])
        np.save(f, fileName)
        f.close()