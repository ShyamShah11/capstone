import sklearn
import sklearn.model_selection
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import ast
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)

gestures={0:"palm", 1:"l-shape"}
f=open("./data/palm.txt", "r")
palm_X = ast.literal_eval(f.read())
f=open("./data/l.txt","r")
l_X = ast.literal_eval(f.read())
X = np.concatenate((palm_X, l_X)) #create array with all image data
y = np.concatenate((np.zeros(len(palm_X)) , np.ones(len(l_X)))) #create array with actual gestures
nsamples, nx, ny = X.shape #reshape array to fit into knn model
X_reshaped = X.reshape((nsamples, nx*ny)) 
print(X_reshaped.shape)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_reshaped, y, test_size = 0.2, random_state=4)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2) #create model with 2 gestures
knn.fit(X_train, y_train) #train model
#y_pred = knn.predict(X_test)
#print (sklearn.metrics.accuracy_score(y_test,y_pred)) #checks accuracy of model
f=open("./testdata/test.txt", "r")
test = np.array(ast.literal_eval(f.read()))
f.close()
test = np.delete(test, list(range(0, test.shape[0], 2)), axis=0)#reshape array to fit into knn model
nx, ny = test.shape 
test_reshaped = test.reshape((nx*ny)) 
test_reshaped = test_reshaped.reshape(1, -1) #contains a single sample

print(test_reshaped.shape)
result = knn.predict(test_reshaped)
print(gestures[result[0]])