# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:20:35 2016

@author: zhihuixie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import zipfile
from sklearn import linear_model
    
def get_matrix(df, features, output):
    """
    transform data from pandas dataframe to matrix
    """
    #add a constant column as coefficient for w0
    df["constant"] = 1.0
    feature_x, output_y = df[features].astype(float), df[output].astype(float)
    return feature_x, output_y
    
def visualization(x, y, preds1, preds2):
     """
     visualize the prediction results
     color real output as blue, prediction as red
     """
     plt.figure()
     plt.plot(x, y, ".", color = "blue")
     plt.plot(x, preds1, "-", color = "red")
     plt.plot(x, preds2, "-", color = "black")

class LassoRegression():
    """
    This class implement polynomial linear regression algorithm  
    """
    def __init__(self, train_feature_x, train_output_y):
        """
        four parameters: as pandas dataframe series
        """
        self.train_feature_x = np.array(train_feature_x)
        self.train_output_y = np.array(train_output_y)
        self.initial_weights = np.zeros(self.train_feature_x.shape[1])
    
    def normalize_features(self, features, normalize = True):
        """
        normalizes columns of a given feature matrix
        """
        if normalize:
            norms = np.linalg.norm(features, axis = 0)
            normalized_features = features/norms
        return normalized_features, norms
        
                
    def prediction(self, x, weights):
        """
        If the features matrix (including a column of 1s for the constant) is 
        stored as a 2D array (or matrix) and the regression weights are stored 
        as a 1D array then the predicted output is just the dot product between 
        the features matrix and the weights (with the weights on the right). 
        Write a function ‘predict_output’ which accepts a 2D array 
        ‘feature_matrix’ and a 1D array ‘weights’ and returns a 1D array 
        ‘predictions’. e.g. in python:
        """
        predictions = np.dot(x, weights)
        return predictions
    
    def lasso_coordinate_descent_step(self, i, weights, l1_penalty, normalize = True):
        """
         implement coordinate descent that minimizes the cost function 
         over a single feature i. Note that the intercept (weight 0) is 
         not regularized.
        """
        if normalize:
            features, norm = self.normalize_features(self.train_feature_x)
        else:
            features = self.train_feature_x
        # compute prediction
        prediction = self.prediction(features, weights)
        errors = self.train_output_y - prediction + np.dot(features[:,i],weights[i])
        # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
        ro_i = np.dot(features[:,i].T, errors)
        #print ro_i
    
        if i == 0: # intercept -- do not regularize
            new_weight_i = ro_i
        elif ro_i < -l1_penalty/2.:
            new_weight_i = ro_i + l1_penalty/2.
        elif ro_i > l1_penalty/2.:
            new_weight_i = ro_i - l1_penalty/2.
        else:
            new_weight_i = 0.
    
        return new_weight_i

    
    def lasso_cyclical_coordinate_descent(self, x, initial_weights = None, \
                              l1_penalty = 1e7, tol =1.0, normalize = True, \
                              n_iters = 5000):
        """
        Each time we scan all the coordinates (features) once, we measure the 
        change in weight for each coordinate. If no coordinate changes by more 
        than a specified threshold, we stop.

        For each iteration:

        As you loop over features in order and perform coordinate descent, 
        measure how much each coordinate changes.
        After the loop, if the maximum change across all coordinates is falls 
        below the tolerance, stop. Otherwise, go back to the previous step.
        Return weights
        """
        # setup initial intercept, slope, iter number and change
        if initial_weights is None:
            weights = self.initial_weights
        else:
            weights = initial_weights
        n_iter = 1
        for _ in range(n_iters):
            change = 0
            for i in range(len(weights)): 
                old_weights_i = weights[i]
                weights[i] = self.lasso_coordinate_descent_step(i = i, \
                                 weights = weights,l1_penalty = l1_penalty, \
                                 normalize = normalize)
                if change < abs(weights[i] - old_weights_i):
                    change = abs(weights[i] - old_weights_i)
            #chcek if converegence
            if change < tol:
                print "Converged at interation %d, change: %.2f" % (n_iter, change)
                break
            n_iter += 1   
        #final prediction
        if normalize:
            x, _ = self.normalize_features(x)
        preds = self.prediction(x, weights)
        return weights, preds
        
                 
if __name__ == "__main__":
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, \
    'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, \
    'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, \
    'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, \
    'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, \
    'sqft_lot':int, 'view':int}
    
    #load data
    zf = zipfile.ZipFile("kc_house_data.csv.zip")
    df = pd.read_csv(zf.open("kc_house_data.csv"), dtype=dtype_dict) 
    #get features
    features = ["constant", "sqft_living", "bedrooms"]
    output = "price"
    example_features, example_output = get_matrix(df, features, output)
    
    model = LassoRegression(example_features, example_output) 
    
    #Q1 and Q2
    initial_weights = np.array([1.,4.,1.])
    new_weights = [1., 0, 0]
    l1s = [1.4e8, 1.64e8, 1.73e8, 1.9e8, 2.3e8]
    for l1 in l1s:
        for i in range(1,3):
            new_weights[i] = model.lasso_coordinate_descent_step(i, \
                                  weights = initial_weights, l1_penalty = l1)
        if new_weights[1] != 0 and new_weights[2] == 0:
            print "L1 with weigh1 as nonzero and weight2 as zero:", l1
        if new_weights[1] == 0 and new_weights[2] == 0:
            print "L1 with weigh1 and weight2 as zeros:", l1 
    print "\n"

    
    #Q3:What is the RSS of the learned model on the normalized dataset?
    weights, preds = model.lasso_cyclical_coordinate_descent(x = example_features, \
                              l1_penalty = 1e7, tol =1.0)
    rss = sum((example_output - preds)**2)
    print "The RSS of the learned model on the normalized dataset:", rss, "\n"
    
    #Q4: Which features had weight zero at convergence?
    features_weights = zip(features, weights)
    zero_features = [features_weight for features_weight in features_weights if \
                     features_weight[1] == 0]
    print "features had weight zero at convergence:", zero_features, "\n"
                              
    
    #Q5-Q8:
    #training data
    zf1 = zipfile.ZipFile("kc_house_train_data.csv.zip")
    df_train = pd.read_csv(zf1.open("kc_house_train_data.csv"), dtype=dtype_dict) 
    #test data
    zf2 = zipfile.ZipFile("kc_house_test_data.csv.zip")
    df_test = pd.read_csv(zf2.open("kc_house_test_data.csv"), dtype=dtype_dict)
    #featureas
    features = ["constant", "bedrooms","bathrooms","sqft_living","sqft_lot", \
                "floors","waterfront","view","condition","grade","sqft_above",\
                "sqft_basement","yr_built","yr_renovated"]
    output = "price"
    train_features, train_output = get_matrix(df_train, features, output)
    test_features, test_output = get_matrix(df_train, features, output)
    #fit model
    model = LassoRegression(train_features, train_output) 
    #Q5:In the model trained with l1_penalty=1e7, which of the following features
    #has non-zero weight? (Select all that apply)
    weights1e7, preds = model.lasso_cyclical_coordinate_descent(x = train_features, \
                              l1_penalty = 1e7, tol =1.0)
    features_weights = zip(features, weights1e7)
    non_zero_features = [features_weight for features_weight in features_weights if \
                     features_weight[1] != 0]
    print "features had weight nonzero at convergence with 1e7:", non_zero_features, "\n"
    
    #Q6:In the model trained with l1_penalty=1e8, which of the following features
    #has non-zero weight
    weights1e8, preds = model.lasso_cyclical_coordinate_descent(x = train_features, \
                              l1_penalty = 1e8, tol =1.0)
    features_weights = zip(features, weights1e7)
    non_zero_features = [features_weight for features_weight in features_weights if \
                     features_weight[1] != 0]
    print "features had weight nonzero at convergence with 1e8:", non_zero_features, "\n"
    
    #Q7:In the model trained with l1_penalty=1e4, which of the following features
    #has non-zero weight? (Select all that apply)
    weights1e4, preds = model.lasso_cyclical_coordinate_descent(x = train_features, \
                              l1_penalty = 1e4, tol =5e5)
    features_weights = zip(features, weights1e7)
    non_zero_features = [features_weight for features_weight in features_weights if \
                     features_weight[1] != 0]
    print "features had weight nonzero at convergence with 1e4:", non_zero_features, "\n"
    
    #Q8:Which of the three models gives the lowest RSS on the TEST data?
    _, preds4 = model.lasso_cyclical_coordinate_descent(x = test_features, \
                              l1_penalty = 1e4, tol =1.0)
    _, preds7 = model.lasso_cyclical_coordinate_descent(x = test_features, \
                              l1_penalty = 1e7, tol =1.0)
    _, preds8 = model.lasso_cyclical_coordinate_descent(x = test_features, \
                              l1_penalty = 1e8, tol =1.0) 
    rss4 = sum((preds4-test_output)**2)
    rss7 = sum((preds7-test_output)**2)
    rss8 = sum((preds8-test_output)**2)
    total_rss = zip(["rss4", "rss7", "rss8"], [rss4, rss7, rss8])
    print total_rss
    print "The lowest RSS on the TEST data:", min(total_rss, key = lambda x: x[1])
    
    