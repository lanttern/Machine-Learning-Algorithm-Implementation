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

class RidgeRegression():
    """
    This class implement polynomial linear regression algorithm  
    """
    def __init__(self, train_feature_x, train_output_y):
        """
        four parameters: as pandas dataframe series
        """
        self.train_feature_x = train_feature_x
        self.train_output_y = train_output_y
        self.initial_weights = np.zeros(self.train_feature_x.shape[1])
    
    def feature_derivative_ridge(self, errors, weights, l2_penalty, \
                                 feature_is_contstant=False):
        """
         The derivative for the weight for feature i is the 
         sum (over data points) of 2 times the product of the error and the 
         feature itself, plus 2*l2_penalty*w[i].
         IMPORTANT: We will not regularize the constant. 
         Thus, in the case of the constant, the derivative is just twice the 
         sum of the errors (without the 2*l2_penalty*w[0] term).
        """
        
        if feature_is_contstant:
            derivative = 2*(l2_penalty*weights + np.dot(self.train_feature_x.T, errors))
        else:
            weight0 = np.zeros(1)
            if weights.shape[0] == 2:
                weight1 = np.array([weights[1]])
            else: weight1 = weights[1:]
            derivative = 2*(np.concatenate((weight0,l2_penalty*weight1), axis = 0) \
                         + np.dot(self.train_feature_x.T, errors))
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
                         step_size = 7e-12, tol = 2.5e+7, l2_penalty = 0,\
                         n_iters = 1000, feature_is_contstant=False):
        """
        A gradient descent function using your derivative function above. 
        For each step in the gradient descent, we update the weight for each 
        feature before computing our stopping criteria. The function will take 
        the following parameters:
        2D feature matrix
        array of output values
        initial weights
        step size
        L2 penalty
        maximum number of iterations
        """
        # setup initial intercept, slope, iter number and rss
        if initial_weights is None:
            weights = self.initial_weights
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
            derivative = self.feature_derivative_ridge(errors, weights, l2_penalty, \
                         feature_is_contstant)
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
    

                 
if __name__ == "__main__":
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, \
    'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, \
    'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, \
    'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, \
    'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, \
    'sqft_lot':int, 'view':int}
    """
    #load data
    zf = zipfile.ZipFile("kc_house_data.csv.zip")
    df = pd.read_csv(zf.open("kc_house_data.csv"), dtype=dtype_dict) 
    #get features
    features = ["constant", "sqft_living"]
    output = "price"
    example_features, example_output = get_matrix(df, features, output)
    
    model = RidgeRegression(example_features, example_output) 
    #test feature derivative function
    my_weights = np.array([1., 10.])
    test_predictions = model.prediction(example_features, my_weights)
    errors = test_predictions - example_output # prediction errors
    # next two lines should print the same values
    print model.feature_derivative_ridge(errors=errors, weights = my_weights, \
                                   l2_penalty = 1)
    print np.dot(example_features.T, errors)*2+np.array([0., 20.])
    print ''
    
    """
    #training data
    zf1 = zipfile.ZipFile("kc_house_train_data.csv.zip")
    df_train = pd.read_csv(zf1.open("kc_house_train_data.csv"), dtype=dtype_dict) 
    #test data
    zf2 = zipfile.ZipFile("kc_house_test_data.csv.zip")
    df_test = pd.read_csv(zf2.open("kc_house_test_data.csv"), dtype=dtype_dict)
    
    #simple_weights_0_penalty
    #simple_features = ["constant", "sqft_living"]
    simple_features = ["constant", "sqft_living", "sqft_living15"]
    output = "price"
    simple_feature_train, output_train = get_matrix(df_train, simple_features, output)
    simple_test_feature_test, output_test = get_matrix(df_test, simple_features, output)
    step_size = 1e-12
    max_iterations = 1000
    model_simple_feature = RidgeRegression(simple_feature_train, output_train) 
    preds_low = model_simple_feature.gradient_model(x = simple_feature_train, \
                             step_size = step_size, n_iters = max_iterations)
                                                
    #simple_weights_high_penalty
    l2 = 1e11
    preds_high = model_simple_feature.gradient_model(x = simple_feature_train, \
                             step_size = step_size, n_iters = max_iterations, \
                             l2_penalty =l2)
    #visualization                                  
    visualization(df_train["sqft_living"], df_train["price"], preds_low, preds_high)
    
    simple_weights_0_penalty = np.array([-1.63113515e-01, 2.63024369e+02])
    simple_weights_high_penalty = np.array([9.76730381, 124.57217567])
    
    #compute rss:
    #weights with zeros
    print "Weight0.........."
    preds_zero_test = preds_low_test = model_simple_feature.gradient_model(x = simple_test_feature_test, \
                             step_size = step_size, n_iters = 1)
    rss_zero_test = sum((output_test-preds_zero_test)**2)
    print "Rss for zero weights test: %E" %rss_zero_test
    price_error_zero = output_test[0] - preds_zero_test[0]    
    print "Error for zero test: %E" %price_error_zero, "\n"
    #weight with low pentaly
    print "Low weights.........."
    preds_low_test = model_simple_feature.gradient_model(x = simple_test_feature_test, \
                             step_size = step_size, n_iters = max_iterations)
    rss_low_test = sum((output_test-preds_low_test)**2)
    print "Rss for low test: %E" %rss_low_test   
    price_error_low = output_test[0] - preds_low_test[0]    
    print "Error for low test: %E" %price_error_low, "\n"
    #weights with high pentaly
    print "High weights.........."                  
    preds_high_test = model_simple_feature.gradient_model(x = simple_test_feature_test, \
                             step_size = step_size, n_iters = max_iterations, \
                             l2_penalty =l2)
    rss_high_test = sum((output_test-preds_high_test)**2)
    print "Rss for high test: %E" %rss_high_test
    price_error_high = output_test[0] - preds_high_test[0]    
    print "Error for high test: %E" %price_error_high, "\n"
    
    