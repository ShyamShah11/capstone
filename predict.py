import sklearn.neighbors
import numpy as np
import pickle
import sys
import ast

np.set_printoptions(threshold=sys.maxsize)

gestures={0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c", 7:"none", 8:"none"}

f=open("./testdata/test.txt", "r")
test = np.array(ast.literal_eval(f.read()))
f.close()
test = np.delete(test, list(range(0, test.shape[0], 2)), axis=0)#reshape array to fit into knn model
nx, ny = test.shape 
test_reshaped = test.reshape((nx*ny)) 
test_reshaped = test_reshaped.reshape(1, -1) #contains a single sample

loaded_knn = pickle.load(open("trainedmodel.sav", "rb"))
result = loaded_knn.predict(test_reshaped)
print(gestures[result[0]])
print(loaded_knn.predict_proba(test_reshaped))