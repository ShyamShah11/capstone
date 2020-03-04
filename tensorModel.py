#reference: https://www.tensorflow.org/tutorials/images/cnn
#https://www.tensorflow.org/tutorials/images/classification
#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import sys
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import ast
import cv2

np.set_printoptions(threshold=sys.maxsize)

gestures={0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c", 7:"none", 8:"none"}


f=open("./data/white","rb")
white_X = np.load(f)
#white_X = np.delete(white_X, list(range(0, len(white_X[0]), 2)), axis=1)#reshape image
f=open("./data/palm", "rb")
palm_X = np.load(f)
f=open("./data/l","rb")
l_X = np.load(f)
f=open("./data/fist","rb")
fist_X = np.load(f)
f=open("./data/thumb","rb")
thumb_X = np.load(f)
f=open("./data/index","rb")
index_X = np.load(f)
f=open("./data/ok","rb")
ok_X = np.load(f)
f=open("./data/c","rb")
c_X = np.load(f)
X = np.concatenate((palm_X, l_X, fist_X, thumb_X, index_X, ok_X, c_X, white_X)) #create array with all image data
#X = np.concatenate((palm_X, l_X, fist_X, thumb_X, index_X, ok_X, c_X)) #create array with all image data
print(X.shape) #(700, 240, 640)
y = np.concatenate((np.zeros(len(palm_X)) , np.ones(len(l_X)), np.ones(len(l_X))*2, np.ones(len(l_X))*3, np.ones(len(l_X))*4, np.ones(len(l_X))*5, np.ones(len(l_X))*6, np.ones(len(white_X))*7)) #create array with actual gestures
#y = np.concatenate((np.zeros(len(palm_X)) , np.ones(len(l_X)), np.ones(len(l_X))*2, np.ones(len(l_X))*3, np.ones(len(l_X))*4, np.ones(len(l_X))*5, np.ones(len(l_X))*6)) #create array with actual gestures
print (y.shape)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=4)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0




plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(gestures[y_train[i]])
plt.show()


model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(240, 640)))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8)) #number of gestures

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=3, 
                    validation_data=(X_test, y_test))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print(test_acc)


f=open("./testdata/test.txt", "r")
test = np.array(ast.literal_eval(f.read()))
f.close()
#next three lines makes a regular picture taken from cv2 into a 240x300 image for the model
#test = np.delete(test, list(range(0, test.shape[0], 2)), axis=0)#reshape array to fit into knn model by removing every other row
#print (np.shape(test))
test = np.delete(test, [i for i in range(120)], axis=0) #delete first quarter of image (height)
test = np.delete(test, [i+240 for i in range(120)], axis=0) #delete last quarter of image (height)
print (np.shape(test))
cv2.imwrite("./testdata/test.png", test)
nx, ny = test.shape 
test = test.reshape(1, nx, ny)
test = test/ 255.0 #normalize test
#test_reshaped = test.reshape((nx*ny)) 
#test_reshaped = test_reshaped.reshape(1, -1) #contains a single sample

test = tf.cast(test, tf.float32)
#result = model.predict(test)
#print(result)
#print(gestures[np.argmax(result[0])])
predict_dataset = tf.convert_to_tensor(test)
# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
  print (logits)
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = gestures[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))