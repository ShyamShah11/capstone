import os
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras import datasets, layers, models
import numpy as np 
import logging
logging.getLogger('tensorflow').disabled = True
def predict(img):
    IMG_SIZE = 150
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
    model.add(layers.Dense(2, activation='softmax'))


    model.load_weights("./checkpoints/chk.ckpt")

    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    cv2.imwrite("./testdata/test.png", img)
    img = np.array(img, dtype = 'float16')
    img = img.reshape((1, IMG_SIZE, IMG_SIZE, 1)) #for 2D model
    result = model.predict(img)
    print(result)


    lookup = dict()
    count = 0
    for j in os.listdir('./gestures/leapGestRecog/00/'):
        if not j.startswith('.'): # If running this code locally, this is to 
                                # ensure you aren't reading in hidden folders
            lookup[count] = j
            count = count + 1

    #print (lookup)
    return (lookup[np.argmax(result[0])]) #get the resulting gesture

if __name__ == "__main__":
    test = cv2.imread("./testdata/test.png")
    test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY) #convert to gray
    test = 255-test
    print(predict(test))