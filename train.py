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


#f=open("./data/white.txt","r")
#white_X = ast.literal_eval(f.read())
#white_X = np.delete(white_X, list(range(0, len(white_X[0]), 2)), axis=1)#reshape image
#print ("white_X", len(white_X), len(white_X[0]))
#f=open("./data/black.txt","r")
#black_X = ast.literal_eval(f.read())
#black_X = np.delete(black_X, list(range(0, len(black_X[0]), 2)), axis=1)#reshape image
#print ("black_X", len(black_X), len(black_X[0]))
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
print(X.shape)
y = np.concatenate((np.zeros(len(palm_X)) , np.ones(len(l_X)), np.ones(len(l_X))*2, np.ones(len(l_X))*3, np.ones(len(l_X))*4, np.ones(len(l_X))*5, np.ones(len(l_X))*6)) #create array with actual gestures
nsamples, nx, ny = X.shape #reshape array to fit into knn model
X_reshaped = X.reshape((nsamples, nx*ny)) 
print(X_reshaped.shape)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_reshaped, y, test_size = 0.2, random_state=4)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4, weights="distance") #create model with 2 gestures
knn.fit(X_train, y_train) #train model

#for testing purposes
'''
best_method = ""
best_score = 0.0
best_k = -1
for method in ["uniform", "distance"] :
    for k in range(1, 11) :
        knn_clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        scoretrain = knn_clf.score(X_train, y_train)
        print ("k: " + str(k) + " method: " + method + " test score: " + str(score) + " train score: " + str(scoretrain) + "\n")
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method
 
print("best_k = " + str(best_k))
print("best_score = " + str(best_score))
print("best_method = " + best_method)
'''

pickle.dump(knn, open("trainedmodel.sav", "wb"))