from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import sklearn.neighbors
import numpy as np
import pickle
import sys
import ast
import queue
import predict


def main():
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

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        #record the first frame for checking movement
        if(frameCount == 0):
            frameCheck = frame[gestureArea[0][1]:gestureArea[1][1], gestureArea[0][0]:gestureArea[1][0]]
        frameCount += 1
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

        #Assuming camera records at 60 fps, check if current recording has remained mostly the same for a second
        #then send that image for processing assuming, an image is not currently being processed
        #then update the image to check against
        if (frameCount % 60 == 0):
            if frameQ.empty() and ssim(cv.cvtColor(frameCheck, cv.COLOR_BGR2GRAY), cv.cvtColor(roiFrame, cv.COLOR_BGR2GRAY))<=0.9:
                frameQ.put(thresh)
                return(predict.predict(thresh))
                frameQ.get()
            frameCount = 1
            frameCheck = roiFrame
        
        keyboard = cv.waitKey(10)
        if keyboard == ord('q') or keyboard == 27:
            break
        

if __name__ == "__main__":
    (prob, gest) = main()
    print (prob, gest)