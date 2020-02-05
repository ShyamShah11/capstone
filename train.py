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

gestures={0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c"}

f=open("./data/palm.txt", "r")
palm_X = ast.literal_eval(f.read())
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

X = np.concatenate((palm_X, l_X, fist_X, thumb_X, index_X, ok_X, c_X)) #create array with all image data
y = np.concatenate((np.zeros(len(palm_X)) , np.ones(len(l_X)), np.ones(len(l_X))*2, np.ones(len(l_X))*3, np.ones(len(l_X))*4, np.ones(len(l_X))*5, np.ones(len(l_X))*6)) #create array with actual gestures
nsamples, nx, ny = X.shape #reshape array to fit into knn model
X_reshaped = X.reshape((nsamples, nx*ny)) 
print(X_reshaped.shape)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_reshaped, y, test_size = 0.2, random_state=4)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7) #create model with 2 gestures
knn.fit(X_train, y_train) #train model
#y_pred = knn.predict(X_test)
#print (sklearn.metrics.accuracy_score(y_test,y_pred)) #checks accuracy of model

pickle.dump(knn, open("trainedmodel.sav", "wb"))