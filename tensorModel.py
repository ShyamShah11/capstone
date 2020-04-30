#reference: https://www.kaggle.com/kageyama/keras-hand-gesture-recognition-cnn
#USE THIS TO TRAIN BRAND NEW MODEL WITH 3 INITIAL GESTURES. 
from __future__ import absolute_import, division, print_function, unicode_literals
import tempfile
import tensorflow as tf
import sys
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import ast
import cv2
import os
import random as rn
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json 

#used for debugging
np.set_printoptions(threshold=sys.maxsize)

#create dictionary to map gestures to an index using dataset
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('./gestures/leapGestRecog/00/'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1


x_data = []
y_data = []
IMG_SIZE = 50 #size of image to be used during training
num_images = 15 #number of images per folder (10 folders each)
datacount = 0 #we'll use this to tally how many images are in our dataset
for i in range(0, 10): #loop over the ten top-level folders
    for j in os.listdir('./gestures/leapGestRecog/0' + str(i) + '/'):
        if (not j.startswith('.') and ("01_palm" in j or "02_l" in j or "03_fist" in j)): #again avoid hidden folders, change condition here
            count = 0 #to tally images of a given gesture
            for k in os.listdir('./gestures/leapGestRecog/0' + str(i) + '/' + j + '/'): #loop over the images
                if count >= num_images:
                    break
                path = './gestures/leapGestRecog/0' + str(i) + '/' + j + '/' + k
                img = cv2.imread(path)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray to make training easier
                l, w = img.shape
                img = img[int(l/4):int(l-(l/4)), int(w/4):int(w-(w/4))] #make the images more focused on the hand
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) #shrink size of image
                x_data.append(img)
                category = path.split("/")[5] #extract label from filename
                label = int(category.split("_")[2])-1 #get their indexes based on filename
                y_data.append(label) 
                count = count + 1
            datacount = datacount + count


#show some of the images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_data[i], cmap=plt.cm.binary)
    plt.xlabel(y_data[i])
#plt.show()

#put everything into numpy arrays
x_data = np.array(x_data, dtype = 'float16')
x_data = x_data.reshape(datacount, IMG_SIZE, IMG_SIZE, 1) # needed to reshape so CNN knows its diff images
y_data = np.array(y_data)

#save settings into json file
(label) = np.unique(y_data)
info = {}
info['numclasses'] = len(label)
info['labels'] = []
for i in range (len(label)):
    info['labels'].append({
        'name' : reverselookup[label[i]],
        'index' : int(label[i])
    })

with open('nn_settings.json', 'w') as outfile:
    json.dump(info, outfile)

#split the dataset into test/train
train_split =0.3
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = train_split, random_state = 42)


#construction of model
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
#configures the model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#configure checkpoints to save model weights
checkpoint_path = "./checkpoints/chk.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
#trains the model for a given number of epochs (iterations on a dataset) and validates it.
model.fit(x_train, y_train, epochs=12, batch_size=64, verbose=2, validation_data=(x_test, y_test),  callbacks=[cp_callback])

#display some metrics for testing purposes
test_loss, test_acc = model.evaluate(x_test, y_test)
#print('Test accuracy: {:2.2f}%'.format(test_acc*100))

predictions = model.predict(x_test) # Make predictions towards the test set
y_pred = np.argmax(predictions, axis=1) # Transform predictions into 1-D array with label number
#print (y_pred, y_test)

#print (tf.math.confusion_matrix(y_test,y_pred))
