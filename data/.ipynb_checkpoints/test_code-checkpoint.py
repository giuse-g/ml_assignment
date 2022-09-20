import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\Vincent Lau\Google Drive\Quantitative finance\Machine learning\ml_assignment\data\train.csv",delimiter=";")

# %%

#Split the data set up: 60% training and 40% test -->row 1-4583 for training and rest for test
training_df_x = df.iloc[0:4583,:]