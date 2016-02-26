# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:20:35 2016

@author: zhihuixie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class SimpleRegression():
    """
    This class implement simple linear regression algorithm with one feature 
    and one prediction. 
    """
    def __init__(self, train_feature_x, train_output_y):
        """
        four parameters: as pandas dataframe series
        """
        self.train_feature_x = train_feature_x
        self.train_output_y = train_output_y

    def closed_form_model (self):
        """
        this model calculate intercept and slope based on closed form solution
        numerator = (sum of X*Y) - (1/N)*((sum of X) * (sum of Y))
        denominator = (sum of X^2) - (1/N)*((sum of X) * (sum of X))
        or 
        numerator = (mean of X * Y) - (mean of X)*(mean of Y)
        denominator = (mean of X^2) - (mean of X)*(mean of X)
        
        slope =numerator/denominator
        intercept = (mean of Y) - slope * (mean of X)
        """
        # compute the dimension of features
        nrows = self.train_feature_x.shape[0]
        # compute the sum of input_feature and output
        sum_of_x = self.train_feature_x.sum()
        sum_of_y = self.train_output_y.sum()
        # compute the product of the output and the input_feature and its sum
        sum_of_xy = np.dot(self.train_feature_x, self.train_output_y)
        # compute the squared value of the input_feature and its sum
        sum_of_squared_x = sum(self.train_feature_x**2)
        # use the formula for the slope
        numerator = sum_of_xy - (sum_of_x*sum_of_y)/nrows
        denominator = sum_of_squared_x - (sum_of_x*sum_of_x)/nrows
        slope = numerator/denominator
        # use the formula for the intercept
        intercept = self.train_output_y.mean() - slope*self.train_feature_x.mean()  
        return (intercept, slope)
        
    def gradient_model (self, initial_inter = 0, initial_slope = 0, \
                         step_size = 7e-12, tol = 2.5e7, n_iters = 50000):
        """
        this model calculate intercept and slope based on gradient descent solution
        In each step of the gradient descent we will do the following:

        1. Compute the predicted values given the current slope and intercept

        2. Compute the prediction errors (prediction - Y)

        3. Update the intercept:

           compute the derivative: sum(errors)
           compute the adjustment as step_size times the derivative
           decrease the intercept by the adjustment
        4. Update the slope:

           compute the derivative: sum(errors*input)
           compute the adjustment as step_size times the derivative
           decrease the slope by the adjustment
        5. Compute the magnitude of the gradient

        6. Check for convergence
        """
        # setup initial intercept, slope, iter number and rss
        intercept = initial_inter
        slope = initial_slope
        n_iter = 1
        rss = float("inf")
        for i in range(n_iters):
            # Compute the predicted values given the current slope and intercept
            pred_values = intercept + slope*self.train_feature_x
            # Compute the prediction errors (prediction - Y)
            errors = pred_values - self.train_output_y
            # Compute the RSS
            rss = sum(errors**2)
            #Update the intercept:
            derivative_i = sum(errors)
            intercept = intercept - derivative_i*(step_size)
            #Update the slope
            derivative_s = np.dot(errors, self.train_feature_x)
            slope = slope - 2*derivative_s*(step_size)
        
            gradient_magnitude = math.sqrt(derivative_i**2+derivative_s**2)
            if gradient_magnitude < tol:
                print "\n"
                print "Converaged at interation %d, rss: %E" % (n_iter, rss)
                break 
                
            if math.isinf(rss) or math.isnan(rss) or math.isinf(intercept) \
               or math.isinf(slope):
               print "\n"
               print "The model is invalide, please use a small step_size to train the model."  
               break            
            n_iter += 1
        if gradient_magnitude >= tol:
            print "\n"
            print "Not converaged, stop at interation %d, rss: %E" % (n_iter, rss)
        return (intercept, slope)
    
    def predictions(self, x, default_model = False):
        """
        predict output values for test dataset
        """
        # choose closed-form model
        if not default_model:
            intercept, slope = self.closed_form_model()
        # choose gradient descent model
        else:
            intercept, slope = default_model
        # calculate the predicted values:
        predicted_values = intercept + slope*x
        return predicted_values
    
    def residual_sum_of_squares(self, x, y, default_model = False):
        """
        compute rss for test dataset
        """
        # First get the predictions
        pred_values = self.predictions(x, default_model = default_model)
        # Compute the prediction errors (prediction - Y)
        errors = pred_values - y
        # Compute the RSS
        rss = sum(errors**2)
        return(rss)
    
    def inverse_regression_predictions(self, y, default_model = False):
        """
        compute feature in test dataset
        """
        # choose closed-form model
        if not default_model:
            intercept, slope = self.closed_form_model()
        # choose gradient descent model
        else:
            intercept, slope = default_model
        # solve output = intercept + slope*input_feature for input_feature. 
        # Use this equation to compute the inverse predictions:    
        estimated_feature = (y - intercept)/slope
        return estimated_feature
        
    def visualization(self, x, y, model = False):
        """
        visualize the prediction results
        color real output as blue, prediction as red
        """
        plt.figure()
        plt.plot(x, y, ".", color = "blue")
        plt.plot(x, self.predictions(x, default_model = model),\
                 ".", color = "red")
                 
if __name__ == "__main__":
    import zipfile
    # Read training data
    zf1 = zipfile.ZipFile("kc_house_train_data.csv.zip")
    df_train = pd.read_csv(zf1.open("kc_house_train_data.csv"))
    #print df_train.dtypes
    # Read test data
    zf2 = zipfile.ZipFile("kc_house_test_data.csv.zip")
    df_test = pd.read_csv(zf2.open("kc_house_test_data.csv"))
    #print df_test.head(5)
    
    
    # get sqft feature and output
    train_feature_x, train_output_y, test_feature_x, test_output_y = \
    df_train["sqft_living"].astype(float), df_train["price"].astype(float), \
    df_test["sqft_living"].astype(float), df_test["price"].astype(float)
    
    # train closed-form model
    model = SimpleRegression(train_feature_x, train_output_y)
    model.visualization(test_feature_x, test_output_y)
    
    # train gradient descent model
    grad_model = model.gradient_model()
    model.visualization(test_feature_x, test_output_y, model = grad_model)
    
    """
    #Q1:Using your Slope and Intercept from predicting prices from square feet, 
    #what is the predicted price for a house with 2650 sqft? Use American-style 
    #decimals without comma separators (e.g. 300000.34), and round your answer 
    #to 2 decimal places. Do not include the dollar sign.
    living_sqft = 2650
    price = model.predictions(living_sqft)
    print "\n"
    print "Q1: The price for %d ft is %.2f."%(living_sqft, price), "\n"
    
    #Q2:Using the learned slope and intercept from the squarefeet model,
    #what is the RSS for the simple linear regression using squarefeet 
    #to predict prices on TRAINING data?
    rss = model.residual_sum_of_squares(train_feature_x, train_output_y)
    print "Q2: The RSS for closed-form model prediction is", rss
    print "\n"
    
    #Q3:According to the inverse regression function and the regression slope 
    #and intercept from predicting prices from square-feet, what is the 
    #estimated square-feet for a house costing $800,000?
    cost = 800000
    living_sqft = model.inverse_regression_predictions(cost)
    print "Q3: The living_sqft for %d dollar is %.2f."%(cost, living_sqft), "\n"
    
    #Q4: compare sqft and bedroom feature
    
    # rss of test data using living sqft
    rss_1 = model.residual_sum_of_squares(test_feature_x, test_output_y)
    # get bedroom feature and output
    train_feature_x, train_output_y, test_feature_x, test_output_y = \
    df_train["bedrooms"].astype(float), df_train["price"].astype(float), \
    df_test["bedrooms"].astype(float), df_test["price"].astype(float)
    # train closed-form model
    model2 = SimpleRegression(train_feature_x, train_output_y)
    rss_2 =  model2.residual_sum_of_squares(test_feature_x, test_output_y)    
    model2.visualization(test_feature_x, test_output_y)
    
    # train gradient descent model
    grad_model2 = model2.gradient_model()
    rss_grad2 = model2.residual_sum_of_squares(test_feature_x, test_output_y, default_model = grad_model)
    model2.visualization(test_feature_x, test_output_y, model = grad_model)
    
    print "Q4: The rss for using living sqft is %.2e, for using bedrooms is %.2e"\
           %(rss_1, rss_2), "\n"
    """
    