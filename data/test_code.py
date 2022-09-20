from random import Random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\Vincent Lau\Google Drive\Quantitative finance\Machine learning\ml_assignment\data\train.csv",delimiter=";")
#Convert string to float in 2nd column
df['meaneduc'] = df['meaneduc'].astype(float)
df['overcrowding'] = df['overcrowding'].astype(float)
df['v2a1'] = df['v2a1'].astype(float)

print(df.info())
#Split the data set up: 60% training and 40% test -->row 1-4583 for training and rest for test
training_df_x = df.iloc[0:4583,1:] #Training set features
training_df_y = df.iloc[0:4583,0] #Training set classification


n_estimators = 100 #Amount of trees
criterion = "gini"  #Criterion: gini / entropy / log_loss
max_depth = 10 # Max depth of tree
min_samples_split = 5 # Min samples required to split an internal node
min_samples_leaf = 5 # Min samples required at leaf node


clf = RandomForestClassifier(n_estimators, criterion, max_depth)
#clf.fit(training_df_x, training_df_y)