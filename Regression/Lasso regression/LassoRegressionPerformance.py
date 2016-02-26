# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 22:19:31 2016

@author: zhihuixie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from zipfile import ZipFile
from math import sqrt, log

def add_features(df):
    """
    add new features to pandas datafram df
    """
    df['sqft_living_sqrt'] = df['sqft_living'].apply(sqrt)
    df['sqft_lot_sqrt'] = df['sqft_lot'].apply(sqrt)
    df['bedrooms_square'] = df['bedrooms']*df['bedrooms']
    df['floors_square'] = df['floors']*df['floors']
    
    return df
        
if __name__ == "__main__":
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, \
                  'sqft_living15':float, 'grade':int, 'yr_renovated':int, \
                  'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, \
                  'sqft_lot15':float, 'sqft_living':float, 'floors':float, \
                  'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,\
                  'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
                  
                  
    all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

    #Q1:Which features have been chosen by LASSO?    
    # read dataset
    zf = ZipFile('kc_house_data.csv.zip')
    sales = pd.read_csv(zf.open('kc_house_data.csv'), dtype=dtype_dict)
    #add features
    df = add_features(sales)
    # set parameters
    model_all = linear_model.Lasso(alpha=5e2, normalize=True) 
    # learn weights
    model_all.fit(df[all_features], df["price"]) 
    # output features with zero coef
    coefs = model_all.coef_
    feature_coefs = zip(all_features, coefs)
    print "Q1:Features with nonzero coefs: ", [feature_coef for feature_coef in \
                                 feature_coefs if feature_coef[1] != 0], "\n" 
    #Q2:Which was the best value for the l1_penalty, i.e. which value of 
    #l1_penalty produced the lowest RSS on VALIDATION data?
    #read data
    zf1 = ZipFile('wk3_kc_house_valid_data.csv.zip')
    zf2 = ZipFile('wk3_kc_house_train_data.csv.zip')
    zf3 = ZipFile('wk3_kc_house_test_data.csv.zip')
    valid = pd.read_csv(zf1.open('wk3_kc_house_valid_data.csv'), dtype=dtype_dict)
    train = pd.read_csv(zf2.open('wk3_kc_house_train_data.csv'), dtype=dtype_dict)
    test = pd.read_csv(zf3.open('wk3_kc_house_test_data.csv'), dtype=dtype_dict) 
    #add features
    valid_all = add_features(valid)            
    train_all = add_features(train)
    test_all = add_features(test)  
    #initiate rss
    best_rss = float("inf")   
    best_l1 = 0       
    # test l1 penality
    l1_penalities = np.logspace(1,7, num = 13)   
    for l1_penality in l1_penalities:
         model = linear_model.Lasso(alpha = l1_penality, normalize = True)
         model.fit(train_all[all_features], train_all["price"])
         preds = model.predict(valid_all[all_features])
         temp_rss = sum((preds - valid_all["price"])**2)      
         if temp_rss < best_rss:
            best_rss = temp_rss   
            best_l1 = l1_penality
    print "Q2: the best penality is:", best_l1, "\n"
    #Q3: Using the best L1 penalty, how many nonzero weights do you have? 
    #Count the number of nonzero coefficients first, and add 1 if the intercept 
    #is also nonzero. A succinct way to do this is
    model = linear_model.Lasso(alpha = best_l1, normalize = True)
    model.fit(train_all[all_features], train_all['price'])
    nonzero_coef = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
    print "Q3: number of non zero coefs: ", nonzero_coef, "\n"
    
    #Q4:What values did you find for l1_penalty_min and l1_penalty_max?
    max_nonzeros = 7
    l1_penalty_min = float("-inf")
    min_nonzeros = 7
    l1_penalty_max = float("inf")
    l1_penalities = np.logspace(1, 4, num=20)
    best_rss = float("inf")   
    best_l1 = 0 
    coefs = []
    
    for l1_penality in l1_penalities:
         model = linear_model.Lasso(alpha = l1_penality, normalize = True)
         model.fit(train_all[all_features], train_all["price"])
         nonzero_coef = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
         if nonzero_coef > max_nonzeros:
             max_nonzeros = nonzero_coef
             if l1_penality > l1_penalty_min:
                 l1_penalty_min = l1_penality
         elif nonzero_coef < min_nonzeros:
             min_nonzeros = nonzero_coef
             if l1_penality < l1_penalty_max:
                 l1_penalty_max = l1_penality
    print "Q4: values for l1_penalty_min and l1_penalty_max:", l1_penalty_min, \
           l1_penalty_max, "\n"
                      
    #Q5:What value of l1_penalty in our narrow range has the lowest RSS on 
    #the VALIDATION set and has sparsity equal to ‘max_nonzeros’?
    max_nonzeros = 7                 
    for l1_penality in np.linspace(l1_penalty_min,l1_penalty_max,20):
         model = linear_model.Lasso(alpha = l1_penality, normalize = True)
         model.fit(train_all[all_features], train_all["price"])
         preds = model.predict(valid_all[all_features])
         nonzero_coef = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
         if nonzero_coef == max_nonzeros:
             temp_rss = sum((preds - valid_all["price"])**2)      
             if temp_rss < best_rss:
                best_rss = temp_rss   
                best_l1 = l1_penality
                coefs = model.coef_
    print "Q5: the best penality with %d features is:"%max_nonzeros, best_l1, "\n"
    
    #Q6:What features in this model have non-zero coefficients?
    feature_coefs = zip(all_features, coefs)
    print "Q6:Features with nonzero coefs in trained model: ", [feature_coef for \
          feature_coef in feature_coefs if feature_coef[1] != 0], "\n" 

    

    
    
        
   