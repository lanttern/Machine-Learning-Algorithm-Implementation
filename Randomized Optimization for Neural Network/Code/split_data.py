# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 23:50:03 2016

@author: zhihuixie
"""

from sklearn.cross_validation import train_test_split
import pandas as pd

"""
method to write data to files
"""
def output_file(data, file_name):
    df = pd.DataFrame(data)
    df.to_csv("../Dataset/" + file_name, header = False, index = False)

# split dataset 2 - EEG dataset to training, validation and test datasets 
df = pd.read_csv("../Dataset/EEG data.csv")
y = df["Self-defined label"]
df.drop("Self-defined label", inplace = True, axis = 1)
df.drop("predefined label", inplace = True, axis = 1)
df.drop("subject ID", inplace = True, axis = 1)
df.drop("Video ID", inplace = True, axis = 1)
X = df.values
X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size=0.2, 
                                                        random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val, 
                                                  test_size=0.2,random_state=42)
# output data to files
data_names = [(X_train, "X_train.csv"), (X_test, "X_test.csv"),(X_val, "X_val.csv"),
              (y_train, "y_train.csv"),(y_test, "y_test.csv"),(y_val, "y_val.csv"),]

for (data, name) in data_names:
    output_file(data, name)