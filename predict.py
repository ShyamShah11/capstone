import sklearn.neighbors
import numpy as np
import pickle
import sys
import ast
import cv2

np.set_printoptions(threshold=sys.maxsize)

gestures={0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c", 7:"none", 8:"none"}

f=open("./testdata/test.txt", "r")
test = np.array(ast.literal_eval(f.read()))
f.close()
#next three lines makes a regular picture taken from cv2 into a 240x300 image for the model
#test = np.delete(test, list(range(0, test.shape[0], 2)), axis=0)#reshape array to fit into knn model by removing every other row
print (np.shape(test))
test = np.delete(test, [i for i in range(120)], axis=0) #delete first quarter of image (height)
test = np.delete(test, [i+240 for i in range(120)], axis=0) #delete last quarter of image (height)
test = np.delete(test, [i for i in range(190)], axis=1) #delete first third of image (width)
test = np.delete(test, [i+300 for i in range(150)], axis=1) #delete last third of image (width)
print (np.shape(test))
nx, ny = test.shape 
test_reshaped = test.reshape((nx*ny)) 
test_reshaped = test_reshaped.reshape(1, -1) #contains a single sample

cv2.imwrite("./testdata/test.png", test)

loaded_knn = pickle.load(open("trainedmodel.sav", "rb"))
result = loaded_knn.predict(test_reshaped)
print(gestures[result[0]])
print(loaded_knn.predict_proba(test_reshaped))