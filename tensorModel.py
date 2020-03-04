import numpy as np
import tensorflow as tf
import sys
import ast
import sklearn


np.set_printoptions(threshold=sys.maxsize)
gestures={0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c", 7:"none", 8:"none"}


f=open("./data/white.txt","r")
white_X = ast.literal_eval(f.read())
#white_X = np.delete(white_X, list(range(0, len(white_X[0]), 2)), axis=1)#reshape image
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
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_reshaped, y, test_size = 0.2, random_state=4)
y_train  = y_train.astype(int)
y_test  = y_test.astype(int)
batch_size =len(X_train)

print(X_train.shape, y_train.shape,y_test.shape )
## resclae
#from sklearn.preprocessing import MinMaxScaler
scaler = sklearn.preprocessing.MinMaxScaler()
# Train
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# test
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
feature_columns = [tf.feature_column.numeric_column('x', shape=X_train_scaled.shape[1:])]
X_train_scaled.shape[1:]