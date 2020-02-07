from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import sklearn.neighbors
import numpy as np
import pickle
import sys
import ast
import queue

gestures={0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c"}
np.set_printoptions(threshold=sys.maxsize)

def predict(fgMask):
    test = np.array(fgMask)
    test = np.delete(test, list(range(0, test.shape[0], 2)), axis=0)#reshape array to fit into knn model
    nx, ny = test.shape 
    test_reshaped = test.reshape((nx*ny)) 
    test_reshaped = test_reshaped.reshape(1, -1) #contains a single sample

    loaded_knn = pickle.load(open("trainedmodel.sav", "rb"))
    result = loaded_knn.predict(test_reshaped)
    print(result[0])

backSub = cv.createBackgroundSubtractorMOG2()
#creating camera object
capture = cv.VideoCapture(0)
frameQ = queue.Queue()
frameCount = 0
compareQ = queue.Queue(maxsize = 60)

while True:
        ret, frame = capture.read()
        if frame is None:
            break
        cv.imshow("Frame", frame)
        keyboard = cv.waitKey(10)
        if keyboard == ord('p'):
            background =  frame
            break

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    #update the background
    fgMask = cv.absdiff(frame, background)
    
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    if frameQ.empty() and ssim(cv.cvtColor(background, cv.COLOR_BGR2GRAY), cv.cvtColor(frame, cv.COLOR_BGR2GRAY)) <= 0.9:
           fgMask = cv.cvtColor(fgMask, cv.COLOR_BGR2GRAY)
           frameQ.put(fgMask)
           predict(fgMask)
           frameQ.get()
    keyboard = cv.waitKey(10)
    if keyboard == ord('t') or keyboard == 27:
        break
    