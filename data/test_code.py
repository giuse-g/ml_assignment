from random import Random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\Vincent Lau\Google Drive\Quantitative finance\Machine learning\ml_assignment\data\train.csv",delimiter=";", decimal=",")
#Convert string to float in 2nd column

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