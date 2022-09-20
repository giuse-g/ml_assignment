from random import Random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r"C:\Users\Vincent Lau\Google Drive\Quantitative finance\Machine learning\ml_assignment\data\train.csv",delimiter=";", decimal=",")

#Split the data set up: 60% training and 40% test -->row 1-4583 for training and rest for test
training_df_x = df.iloc[0:4583,1:] #Training set features
training_df_y = df.iloc[0:4583,0] #Training set classification

test_df_x = df.iloc[4584:,1:] #Test set features
test_df_y = df.iloc[4584:,0] #Test set y


clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=10)
clf.fit(training_df_x, training_df_y)


#Predict
clf.predict(test_df_x)
print(clf.score(test_df_x,test_df_y))


#Grid search
max_depth = np.arange(5,50,5)
n_estimators = np.arange(50,500,10)
min_samples_split = np.arange(2,100,2)
min_samples_leaf = np.arange(5,100,5)

param_grid = dict(max_depth = max_depth, n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split)

dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, cv=5)
grid_results = grid.fit(training_df_x, training_df_y)
print(grid_results)