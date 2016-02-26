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

def polynomial_dataframe(feature, degree): 
    # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe["power_1"] = feature
    # first check if degree > 1
    if degree > 1:
    # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = "power_" + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = feature.apply(lambda x: x**degree)
    return poly_dataframe
    
def k_fold_cross_validation(k, l2_penalty, train_valid_shuffled):
    n = len(train_valid_shuffled)
    degree =15
    rss = []
    for i in range(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        valid_data = train_valid_shuffled.iloc[start:end+1,:]
        train_feature = train_valid_shuffled.iloc[0:start,:].\
                        append(train_valid_shuffled.iloc[end+1:n,:])
        poly_df = polynomial_dataframe(train_feature['sqft_living'], degree)
        poly_df_valid = polynomial_dataframe(valid_data['sqft_living'], degree)
        model = linear_model.Ridge(alpha = l2_penalty, normalize = True)
        model.fit(poly_df, train_feature['price'])
        preds = model.predict(poly_df_valid)
        rss.append(sum((valid_data['price'] - preds)**2))
    return np.mean(rss)
        


if __name__ == "__main__":
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, \
                  'sqft_living15':float, 'grade':int, 'yr_renovated':int, \
                  'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, \
                  'sqft_lot15':float, 'sqft_living':float, 'floors':float, \
                  'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,\
                  'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
    #Q1:What’s the learned value for the coefficient of feature power_1?    
    # read dataset
    zf = ZipFile('kc_house_data.csv.zip')
    sales = pd.read_csv(zf.open('kc_house_data.csv'), dtype=dtype_dict)
    sales = sales.sort_values(by = ['sqft_living','price'])
    # get parameters for model
    l2_small_penalty = 1.5e-5
    feature = sales['sqft_living']
    degree = 15
    y = sales['price']
    poly15_data = polynomial_dataframe(feature, degree)
    #fit model
    model = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
    model.fit(poly15_data, y)
    preds = model.predict(poly15_data)
    """
    #visualization
    plt.figure()
    plt.plot(feature, y, ".", color = "blue")
    plt.plot(feature, preds, "-", color = "red")
    """
    print model.coef_[0], len(model.coef_)
    #Q2&3: Quiz Question: For the models learned in each of these training sets, 
    #what are the smallest and largest values you learned for the coefficient 
    #of feature power_1? 
    l2_small_penalty = 1e-9
    l2_large_penalty=1.23e2
    coefs_small = []
    coefs_large = []
    for i in range(1,5):
        #generate polynomial features
        zf = ZipFile("wk3_kc_house_set_" + str(i) +"_data.csv.zip")
        df = pd.read_csv(zf.open("wk3_kc_house_set_" + str(i) +"_data.csv"), dtype=dtype_dict)
        df_sorted = df.sort_values(by = ["sqft_living","price"])
        feature = df_sorted["sqft_living"]
        y = df_sorted["price"]
        poly_df = polynomial_dataframe(feature, degree)
        #fit model
        model_s = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
        model_s.fit(poly_df, y)
        coefs_small.append(model_s.coef_[0])
        
        model_l = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
        model_l.fit(poly_df, y)
        coefs_large.append(model_l.coef_[0])
        """
        preds = model.predict(poly_df)

        #visualization
        plt.figure()
        plt.plot(feature, y, ".", color = "blue")
        plt.plot(feature, preds, "-", color = "red")
        """
    print "The coefficients[power_1] with sklearn for dataset with small penalty", coefs_small, "\n"
    print "The coefficients[power_1] with sklearn for dataset with large penalty", coefs_large, "\n"
        
    #Q4:
    #read data
    zf1 = ZipFile('wk3_kc_house_train_valid_shuffled.csv.zip')
    zf2 = ZipFile('wk3_kc_house_train_data.csv.zip')
    zf3 = ZipFile('wk3_kc_house_test_data.csv.zip')
    train_valid_shuffled = pd.read_csv(zf1.open('wk3_kc_house_train_valid_shuffled.csv'), dtype=dtype_dict)
    train = pd.read_csv(zf2.open('wk3_kc_house_train_data.csv'), dtype=dtype_dict)
    test = pd.read_csv(zf3.open('wk3_kc_house_test_data.csv'), dtype=dtype_dict)
    #k-fold cross validation
    k = 10 # 10-fold cross-validation
    l2_penaltys  = np.logspace(3, 9, num=13)
    best_rss = float("inf")
    best_l2 = 0
    for l2_penalty in l2_penaltys:
        averaged_rss = k_fold_cross_validation(k, l2_penalty, train_valid_shuffled)
        if averaged_rss < best_rss:
            best_rss = averaged_rss
            best_l2 = l2_penalty
    print "The best value for the L2 penalty according to 10-fold validation:%f"%best_l2, "\n"
    
   #Q5: the RSS on the TEST data of the model you learn with this L2 penalty?
    poly_df_train = polynomial_dataframe(train['sqft_living'], degree)
    poly_df_test = polynomial_dataframe(test['sqft_living'], degree)
    model = linear_model.Ridge(alpha = best_l2, normalize = True)
    model.fit(poly_df_train, train['price'])
    preds = model.predict(poly_df_test)
    rss = sum((test['price'] - preds)**2)
    print "The RSS on the TEST data of the model you learn with this L2 penalty:%E"%rss


    
    
        
   