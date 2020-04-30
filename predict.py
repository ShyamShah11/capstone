import os
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras import datasets, layers, models
import numpy as np 
import logging
import json
#disable extra tensorflow warning messages
logging.getLogger('tensorflow').disabled = True

def predict(img):
    #need to recreate a network with the same structure (weights do not matter yet)
    IMG_SIZE = 50
    num_classes = 1
    #get information from json
    lookup = dict()
    with open('nn_settings.json') as json_file:
        data = json.load(json_file)
        num_classes = data['numclasses']
        for p in data['labels']:
            lookup[p['index']] = p['name']
    #create model without weights
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    #load in weights from most recent checkpoint
    model.load_weights("./checkpoints/chk.ckpt")

    #format test image to match network specifications
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    cv2.imwrite("./testdata/test.png", img)
    img = np.array(img, dtype = 'float16')
    img = img.reshape((1, IMG_SIZE, IMG_SIZE, 1)) #for 2D model
    #predict what is in the image
    result = model.predict(img)
    print(result)

    #only return a gesture if probability was above 85%
    if max(result[0]) < 0.85:
        return ("none", 0)
    else:
        return (lookup[np.argmax(result[0])], max(result[0])) #get the resulting gesture

if __name__ == "__main__":
    test = cv2.imread("./testdata/test.png")
    test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY) #convert to gray
    test = 255-test
    print(predict(test))