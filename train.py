import sklearn
import sklearn.model_selection
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import ast
f=open("./data/palm.txt", "r")
palm_X = ast.literal_eval(f.read())
f=open("./data/l.txt","r")
l_X = ast.literal_eval(f.read())
X = np.concatenate((palm_X, l_X))
y = np.concatenate((np.zeros(len(palm_X)) , np.ones(len(l_X))))
print(y)
nsamples, nx, ny = X.shape
X_reshaped = X.reshape((nsamples, nx*ny)) 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_reshaped, y, test_size = 0.2, random_state=4)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print (sklearn.metrics.accuracy_score(y_test,y_pred))