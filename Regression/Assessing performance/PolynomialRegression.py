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

class PolynomialRegression():
    """
    This class implement polynomial linear regression algorithm  
    """
    def __init__(self, train_feature_x, train_output_y):
        """
        four parameters: as pandas dataframe series
        """
        self.train_feature_x = train_feature_x
        self.train_output_y = train_output_y
        self.initial_weight = np.zeros(self.train_feature_x.shape[1])
    
    def feature_derivative(self, errors):
        """
         If we have a the values of a single input feature in an array ‘feature’ 
         and the prediction ‘errors’ (predictions - output) then the derivative 
         of the regression cost function with respect to the weight of ‘feature’ 
         is just twice the dot product between ‘feature’ and ‘errors’. Write a 
         function that accepts a ‘feature’ array and ‘error’ array and returns 
         the ‘derivative’ (a single number). e.g. in python
        """
        derivative = 2*np.dot(self.train_feature_x.T, errors)
        return derivative
        
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
        
    def gradient_model (self, x, initial_weights = None, \
                         step_size = 7e-12, tol = 2.5e+7, n_iters = 1000):
        """
        This model calculate intercept and slope based on gradient descent solution
        In each step of the gradient descent we will do the following:

        1. Accepts a numpy feature_matrix 2D array, a 1D output array, an array 
        of initial weights, a step size and a convergence tolerance.
        
        2. While not converged updates each feature weight by subtracting the 
        step size times the derivative for that feature given the current weights
        
        3. At each step computes the magnitude/length of the gradient (square 
        root of the sum of squared components)
        
        4. When the magnitude of the gradient is smaller than the input 
        tolerance returns the final weight vector.
        """
        # setup initial intercept, slope, iter number and rss
        if initial_weights is None:
            weights = self.initial_weight
        else:
            weights = initial_weights
        n_iter = 0
        rss = float("inf")
        for i in range(n_iters):
            # Compute the predicted values given the current slope and intercept
            pred_values = self.prediction(self.train_feature_x, weights)
            # Compute the prediction errors (prediction - Y)
            errors = pred_values - self.train_output_y
            # Compute the RSS and gradient_sum_squares
            rss = sum(errors**2)
            #Update the weights:
            derivative = self.feature_derivative(errors)
            weights = weights - derivative * (step_size)        
            gradient_magnitude = math.sqrt(sum(derivative**2))
            #check if converged
            if gradient_magnitude < tol:
                print "\n"
                print "Converaged at interation %d, rss: %E" % (n_iter, rss)
                break 
            # check if given wrong parameters   
            if math.isinf(rss) or math.isnan(rss):
               print "\n"
               print "The model is invalide, please use a small step_size to train the model."  
               break  
            n_iter += 1
        if gradient_magnitude >= tol:
            print "\n"
            print "Not converaged, stop at interation %d, rss: %E" % (n_iter, rss)
        
        #check weights
        print "\n"
        print "The weights for features: ", weights
        #final prediction
        preds = self.prediction(x, weights)
        return preds
    def least_squares_solution(self,x):
        """
        Return the least-squares solution to a linear matrix equation.
        Solves the equation a x = b by computing a vector x that minimizes the 
        Euclidean 2-norm || b - a x ||^2. The equation may be under-, well-, 
        or over- determined (i.e., the number of linearly independent rows of 
        a can be less than, equal to, or greater than its number of linearly 
        independent columns). If a is square and of full rank, then x 
        (but for round-off error) is the “exact” solution of the equation.
        """
        weights, rss, _, _ = np.linalg.lstsq(self.train_feature_x, self.train_output_y)
        print "\n"
        print "The RSS for training dataset is:", rss, "\n"
        print "The weights for training dataset is:", weights 
        preds = self.prediction(x, weights)
        return preds
    
    def visualization(self, x, y, preds):
        """
        visualize the prediction results
        color real output as blue, prediction as red
        """
        plt.figure()
        plt.plot(x, y, ".", color = "blue")
        plt.plot(x, preds, "-", color = "red")
                 
if __name__ == "__main__":
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, \
    'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, \
    'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, \
    'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, \
    'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, \
    'sqft_lot':int, 'view':int}
    m = linear_model.LinearRegression()
    # data for Test case
    """
    zf = zipfile.ZipFile("kc_house_data.csv.zip")
    df = pd.read_csv(zf.open("kc_house_data.csv"), dtype=dtype_dict)
    df_sorted = df.sort_values(by = ["sqft_living","price"])
    """
    """
    #Q1:Is the sign (positive or negative) for power_15 the same in all four models?
    #Q2:The plotted fitted lines all look the same in all four plots    
    degree = 15
    for i in range(1,5):
        #generate polynomial features
        zf = zipfile.ZipFile("wk3_kc_house_set_" + str(i) +"_data.csv.zip")
        df = pd.read_csv(zf.open("wk3_kc_house_set_" + str(i) +"_data.csv"), dtype=dtype_dict)
        df_sorted = df.sort_values(by = ["sqft_living","price"])
        feature = df_sorted["sqft_living"]
        poly_df = polynomial_dataframe(feature, degree)
        # get feature names
        features = list(poly_df.columns.values)
        # add output
        poly_df["price"] = df_sorted["price"]
        output = "price"
        # train gradient descent model
        train_feature_x, train_output_y = get_matrix(poly_df, features, output)
        model = PolynomialRegression(train_feature_x, train_output_y) 
        print "The coefficients for set %d"%i
        preds = model.least_squares_solution(x = train_feature_x)
        model.visualization(train_feature_x, train_output_y, preds)
        #confirm with sklearn algorithm
        m.fit(train_feature_x, train_output_y)
        p = m.predict(train_feature_x)
        model.visualization(train_feature_x, train_output_y, p)
        print "The coefficients with sklearn", m.coef_
    """    
    
    #Q3:Which degree (1, 2, …, 15) had the lowest RSS on Validation data?
    #training data
    zf1 = zipfile.ZipFile("wk3_kc_house_train_data.csv.zip")
    df1 = pd.read_csv(zf1.open("wk3_kc_house_train_data.csv"), dtype=dtype_dict)
    df1 = df1.sort_values(by = ["sqft_living","price"])
    #validation data
    zf2 = zipfile.ZipFile("wk3_kc_house_valid_data.csv.zip")
    df2 = pd.read_csv(zf2.open("wk3_kc_house_valid_data.csv"), dtype=dtype_dict)
    df2 = df2.sort_values(by = ["sqft_living","price"])    
    #test data
    zf3 = zipfile.ZipFile("wk3_kc_house_test_data.csv.zip")
    df3 = pd.read_csv(zf3.open("wk3_kc_house_test_data.csv"), dtype=dtype_dict)
    df3 = df3.sort_values(by = ["sqft_living","price"])
    for degree in range(1,16):
        #generate polynomial features
        feature = df1["sqft_living"]
        poly_df = polynomial_dataframe(feature, degree)
        # get feature names
        features = ["constant"]+list(poly_df.columns.values)
        # add output
        poly_df["price"] = df1["price"]
        
        valid_feature = df2["sqft_living"]
        valid_poly_df = polynomial_dataframe(valid_feature, degree)
        # get feature names
        valid_features = ["constant"]+list(valid_poly_df.columns.values)
        # add output
        valid_poly_df["price"] = df2["price"]
        
        output = "price"
        # train gradient descent model
        train_feature_x, train_output_y = get_matrix(poly_df, features, output)
        valid_feature_x, valid_output_y = get_matrix(valid_poly_df, \
                                          valid_features, output)
        model = PolynomialRegression(train_feature_x, train_output_y) 
        preds = model.least_squares_solution(x = valid_feature_x)
        rss1 = sum((valid_output_y-preds)**2)
        print "The RSS for degree %d is %E"%(degree, rss1)
        
        #confirm with sklearn algorithm
        m.fit(train_feature_x, train_output_y)
        p = m.predict(valid_feature_x)
        rss2 = sum((valid_output_y-p)**2)
        print "The RSS for degree %d with sklearn is %E"%(degree,rss2), "\n"
        model.visualization(valid_feature_x["power_1"], valid_output_y,preds)
    """  
    #Q4: What is the RSS on TEST data for the model with the degree selected from 
    #Validation data? (Make sure you got the correct degree from the previous question)
    feature = df1["sqft_living"]
    poly_df = polynomial_dataframe(feature, 2)
    # get feature names
    features = ["constant"] + list(poly_df.columns.values)
    # add output
    poly_df["price"] = df1["price"]
        
    test_feature = df3["sqft_living"]
    test_poly_df = polynomial_dataframe(test_feature, 2)
    # get feature names
    test_features = list(test_poly_df.columns.values)
    # add output
    test_poly_df["price"] = df3["price"]
        
    output = "price"
    # train gradient descent model
    train_feature_x, train_output_y = get_matrix(poly_df, features, output)
    test_feature_x, test_output_y = get_matrix(test_poly_df, \
                                          test_features, output)
    model = PolynomialRegression(train_feature_x, train_output_y) 
    preds = model.least_squares_solution(x = test_feature_x)
    rss1 = sum((test_output_y-preds)**2)
    print "The RSS for degree %d is %E"%(2, rss1)
        
    #confirm with sklearn algorithm
    m.fit(train_feature_x, train_output_y)
    p = m.predict(test_feature_x)
    rss2 = sum((test_output_y-p)**2)
    print "The RSS for degree %d with sklearn is %E"%(2,rss2), "\n"
    """