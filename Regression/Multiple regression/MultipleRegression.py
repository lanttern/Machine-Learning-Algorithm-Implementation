# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:20:35 2016

@author: zhihuixie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class MultipleRegression():
    """
    This class implement simple linear regression algorithm with one feature 
    and one prediction. 
    """
    def __init__(self, train_feature_x, train_output_y):
        """
        four parameters: as pandas dataframe series
        """
        # add a constant column
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
                         step_size = 7e-12, tol = 2.5e+7, n_iters = 10000):
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
        
    def visualization(self, x, y, preds):
        """
        visualize the prediction results
        color real output as blue, prediction as red
        """
        plt.figure()
        plt.plot(x, y, ".", color = "blue")
        plt.plot(x, preds, ".", color = "red")
                 
if __name__ == "__main__":
    import zipfile
    
    def get_matrix(df, features, output):
        """
        transform data from pandas dataframe to matrix
        """
        #add a constant column as coefficient for w0
        df["constant"] = 1.0
        feature_x, output_y = df[features].astype(float), df[output].astype(float)
        return feature_x, output_y
    # Read training data
    zf1 = zipfile.ZipFile("kc_house_train_data.csv.zip")
    df_train = pd.read_csv(zf1.open("kc_house_train_data.csv"))
    
    # Read test data
    zf2 = zipfile.ZipFile("kc_house_test_data.csv.zip")
    df_test = pd.read_csv(zf2.open("kc_house_test_data.csv"))
    #print df_test.head(5)
    
    """
    #A1-Q1:What is the mean value (arithmetic average) of the 'bedrooms_squared' 
    #feature on TEST data? (round to 2 decimal places)
    mean_bed = np.mean(df_test.bedrooms ** 2)
    print "The mean value (arithmetic average) of the \
          'bedrooms_squared'feature on TEST data: %.2f"%mean_bed, "\n"
          
    #A1-Q2:What is the mean value (arithmetic average) of the 'bed_bath_rooms' 
    #feature on TEST data? (round to 2 decimal places)
    mean_bb = np.mean(df_test.bedrooms *df_test.bathrooms)
    print "The mean value (arithmetic average) of the \
          'bed_bath_rooms'feature on TEST data: %.2f"%mean_bb, "\n"
          
    #A1-Q3:What is the mean value (arithmetic average) of the 'log_sqft_living' 
    #feature on TEST data? (round to 2 decimal places)
    mean_log_sl = np.mean(np.log(df_test.sqft_living))
    print "The mean value (arithmetic average) of the \
          'log_sqft_living'feature on TEST data: %.2f"%mean_log_sl, "\n"
          
    #A1-Q4:What is the mean value (arithmetic average) of the 'lat_plus_long' 
    #feature on TEST data? (round to 2 decimal places)
    mean_lpl = np.mean(df_test.lat + df_test.long)
    print "The mean value (arithmetic average) of the \
          'log_sqft_living'feature on TEST data: %.2f"%mean_lpl, "\n"
          
    #A1-Q5:What is the sign (positive or negative) for the coefficient/weight 
    #for 'bathrooms' in model 1:‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’, and ‘long’?
    
    # get features and output
    features = ["constant", "sqft_living","bedrooms", "bathrooms", "lat", "long"]
    output = "price"
    weights = [-47000,1,1,1,1,1]
    # train gradient descent model
    train_feature_x, train_output_y = get_matrix(df_train, features, output)
    test_feature_x, test_output_y = get_matrix(df_test, features, output)
    model1 = MultipleRegression(train_feature_x, train_output_y)    
    preds1 = model1.gradient_model(x = test_feature_x, initial_weights =weights)

    #A1-Q6:What is the sign (positive or negative) for the coefficient/weight 
    #for 'bathrooms' in model 2:sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,
    #‘long’, and ‘bed_bath_rooms’
    
    # get features and output
    weights = [-47000,1,1,1,1,1,1]
    features = ["constant", "sqft_living","bedrooms", "bathrooms", "lat", \
                "long", "bed_bath_rooms"]
    output = "price"
    df_train["bed_bath_rooms"] = df_train.bedrooms *df_train.bathrooms
    df_test["bed_bath_rooms"] = df_test.bedrooms *df_test.bathrooms
    # train gradient descent model
    train_feature_x, train_output_y = get_matrix(df_train, features, output)
    test_feature_x, test_output_y = get_matrix(df_test, features, output)
    model2 = MultipleRegression(train_feature_x, train_output_y)    
    preds2 = model2.gradient_model(x = test_feature_x, initial_weights = weights)
    
    #A1-Q7:Which model (1, 2 or 3) has lowest RSS on TRAINING Data?
    #Model 3: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, 
    #‘bed_bath_rooms’, ‘bedrooms_squared’, ‘log_sqft_living’, and 
    #‘lat_plus_long’
    # get features and output
    weights = [-47000,1,1,1,1,1,1,1,1,1]
    features = ["constant", "sqft_living","bedrooms", "bathrooms", "lat", \
                "long", "bed_bath_rooms","bedrooms_squared", "log_sqft_living",\
                "lat_plus_long"]
    output = "price"
    df_train["bed_bath_rooms"] = df_train.bedrooms *df_train.bathrooms
    df_train["bedrooms_squared"] = df_train.bedrooms **2
    df_train["log_sqft_living"] = np.log(df_train.sqft_living)
    df_train["lat_plus_long"] = df_train.lat + df_train.long
    df_test["bed_bath_rooms"] = df_test.bedrooms *df_test.bathrooms
    df_test["bedrooms_squared"] = df_test.bedrooms **2
    df_test["log_sqft_living"] = np.log(df_test.sqft_living)
    df_test["lat_plus_long"] = df_test.lat + df_test.long
    # train gradient descent model
    train_feature_x, train_output_y = get_matrix(df_train, features, output)
    test_feature_x, test_output_y = get_matrix(df_test, features, output)
    model3 = MultipleRegression(train_feature_x, train_output_y)    
    preds3 = model3.gradient_model(x = test_feature_x, initial_weights = weights)
    
    #A1-Q8:Which model (1, 2 or 3) has lowest RSS on TESTING Data?
    rss1_test = sum((test_output_y-preds1)**2)
    rss2_test = sum((test_output_y-preds2)**2)
    rss3_test = sum((test_output_y-preds3)**2)
    print "The rss for testing data: %f, %f, %f:" %(rss1_test, rss1_test, rss3_test)
    """
    
    #A2-Q1:What is the value of the weight for sqft_living from your gradient 
    #descent predicting house prices (model 1)?
    #model#1: features: ‘sqft_living’, output: ‘price’, 
    #initial weights: -47000, 1 (intercept, sqft_living respectively),
    #step_size = 7e-12, tolerance = 2.5e7
    features = ["constant", "sqft_living"]
    output = "price"
    weights = np.array([-47000.0,1.0])
    # train gradient descent model
    train_feature_x, train_output_y = get_matrix(df_train, features, output)
    test_feature_x, test_output_y = get_matrix(df_test, features, output)
    model1 = MultipleRegression(train_feature_x, train_output_y)    
    preds1 = model1.gradient_model(x = test_feature_x, initial_weights =weights)
    
    #A2-Q2:What is the predicted price for the 1st house in the TEST data set 
    #for model 1 (round to nearest dollar)?
    print "The predicted price for the 1st house in the TEST data set for model1:\
           %d" %round(preds1[0]), "\n"
    
    #A2-Q3:What is the predicted price for the 1st house in the TEST data set 
    #for model 2 (round to nearest dollar)?
    #model2: 
    features = ["constant", "sqft_living", "sqft_living15"]
    output = "price"
    weights = np.array([-100000, 1, 1])
    step_size = 4e-12
    tol = 1e+9
    
    # train gradient descent model
    train_feature_x, train_output_y = get_matrix(df_train, features, output)
    test_feature_x, test_output_y = get_matrix(df_test, features, output)
    model2 = MultipleRegression(train_feature_x, train_output_y)    
    preds2 = model2.gradient_model(x = test_feature_x, initial_weights = weights,\
                                   step_size = step_size,tol = tol)
    print "The predicted price for the 1st house in the TEST data set for model2:\
           %d" %round(preds2[0]), "\n"
           
    #A2-Q4: Which estimate was closer to the true price for the 1st house on 
    #the TEST data set, model 1 or model 2?
    print "Prediction - real value for model#1: %.1f, for model#2: %.1f" \
          %(test_output_y[0] - preds1[0], test_output_y[0] - preds2[0])
    #A2-Q5: Which model (1 or 2) has lowest RSS on all of the TEST data?
    rss1_test = sum((test_output_y-preds1)**2)
    rss2_test = sum((test_output_y-preds2)**2)
    print "The rss for testing data- model1: %f, model2:%f"%(rss1_test, rss2_test)
