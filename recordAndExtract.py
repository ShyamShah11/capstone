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
gestureArea = []
def predict(fgMask):
    test = np.array(fgMask)
    test = np.delete(test, list(range(0, test.shape[0], 2)), axis=0)#reshape array to fit into knn model
    nx, ny = test.shape 
    test_reshaped = test.reshape((nx*ny)) 
    test_reshaped = test_reshaped.reshape(1, -1) #contains a single sample

    loaded_knn = pickle.load(open("trainedmodel.sav", "rb"))
    result = loaded_knn.predict(test_reshaped)
    print(result[0])

def createBox (event, x,y, flags, params):
    global gestureArea
    if event == cv.EVENT_LBUTTONDOWN:
        gestureArea = [(x, y)]
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        gestureArea.append((x, y))
        # draw a rectangle around the region of interest
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
    if (len(gestureArea)==2):
        cv.rectangle(frameShow, gestureArea[0], gestureArea[1], (0, 255, 0), 1)
    cv.imshow("Frame", frameShow)
    cv.setMouseCallback("Frame", createBox)
    keyboard = cv.waitKey(10)
    if keyboard == ord('p'):
        background =  frame[gestureArea[0][1]:gestureArea[1][1], gestureArea[0][0]:gestureArea[1][0]]
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
        background = cv.GaussianBlur(background,(21,21), 0)
        break

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    dimensions = frame.shape

    roiFrame = frame[gestureArea[0][1]:gestureArea[1][1], gestureArea[0][0]:gestureArea[1][0]]
    fgMask = cv.cvtColor(roiFrame, cv.COLOR_BGR2GRAY)
    fgMask = cv.GaussianBlur(fgMask,(21,21), 0)

    delta = cv.absdiff(fgMask, background)
    trueMask = cv.copyMakeBorder(delta, gestureArea[0][1], dimensions[0]-gestureArea[1][1], gestureArea[0][0], dimensions[1] - gestureArea[1][0],cv.BORDER_CONSTANT,0)

    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', trueMask)
    '''
    if frameQ.empty() and ssim(cv.cvtColor(roiFrame, cv.COLOR_BGR2GRAY), cv.cvtColor(roiMask, cv.COLOR_BGR2GRAY)) >= 0.02:
           fgMask = cv.cvtColor(fgMask, cv.COLOR_BGR2GRAY)
           frameQ.put(roiMask)
           predict(roiMask)
           frameQ.get()
    '''
    keyboard = cv.waitKey(10)
    if keyboard == ord('q') or keyboard == 27:
        cv.imwrite("./testdata/test.png", trueMask)
        break
    