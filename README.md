# capstone

Hand data taken from (not added to repo): https://www.kaggle.com/gti-upm/leapgestrecog/version/1

How it works: 
1. Use record.py to take a picture of your hand
2. Run predict.py and it will output the name of the hand position (currently supports:  0:"palm", 1:"l-shape", 2:"fist", 3:"thumb", 4:"index", 5:"ok", 6:"c")

Use getdata.py to extract more data from Kaggle dataset
Use train.py to feed the algorithm more data
Saved model is not being added to the repo because of a size restriction

Useful resource: https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75