import sklearn
import sklearn.model_selection
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import ast
import cv2
import sys
import pickle

np.set_printoptions(threshold=sys.maxsize)

gestures={0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c", 7:"none", 8:"none"}


f=open("./data/white.txt","r")
white_X = ast.literal_eval(f.read())
white_X = np.delete(white_X, list(range(0, len(white_X[0]), 2)), axis=1)#reshape image
print ("white_X", len(white_X), len(white_X[0]))
f=open("./data/black.txt","r")
black_X = ast.literal_eval(f.read())
black_X = np.delete(black_X, list(range(0, len(black_X[0]), 2)), axis=1)#reshape image
print ("black_X", len(black_X), len(black_X[0]))
f=open("./data/palm.txt", "r")
palm_X = ast.literal_eval(f.read())
print ("palm_X", len(palm_X), len(palm_X[0]))
f=open("./data/l.txt","r")
l_X = ast.literal_eval(f.read())
f=open("./data/fist.txt","r")
fist_X = ast.literal_eval(f.read())
f=open("./data/thumb.txt","r")
thumb_X = ast.literal_eval(f.read())
f=open("./data/index.txt","r")
index_X = ast.literal_eval(f.read())
f=open("./data/ok.txt","r")
ok_X = ast.literal_eval(f.read())
f=open("./data/c.txt","r")
c_X = ast.literal_eval(f.read())


X = np.concatenate((palm_X, l_X, fist_X, thumb_X, index_X, ok_X, c_X, white_X, black_X)) #create array with all image data
print(X.shape)
y = np.concatenate((np.zeros(len(palm_X)) , np.ones(len(l_X)), np.ones(len(l_X))*2, np.ones(len(l_X))*3, np.ones(len(l_X))*4, np.ones(len(l_X))*5, np.ones(len(l_X))*6, np.ones(len(white_X))*7, np.ones(len(black_X))*8)) #create array with actual gestures
nsamples, nx, ny = X.shape #reshape array to fit into knn model
X_reshaped = X.reshape((nsamples, nx*ny)) 
print(X_reshaped.shape)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_reshaped, y, test_size = 0.2, random_state=4)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7) #create model with 2 gestures
knn.fit(X_train, y_train) #train model
#y_pred = knn.predict(X_test)
#print (sklearn.metrics.accuracy_score(y_test,y_pred)) #checks accuracy of model

pickle.dump(knn, open("trainedmodel.sav", "wb"))