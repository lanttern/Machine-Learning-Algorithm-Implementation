# -*- coding: utf-8 -*-
"""
Created on Thu April 22 20:20:35 2016

@author: zhihuixie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import string, json
from zipfile import ZipFile
    
class LogisticRegressionSGA():
    """
    This class implement logistic regression algorithm. 
    """
    def __init__(self, train_feature_x, train_output_y):
        """
        four parameters: as pandas dataframe series
        """
        # add a constant column
        self.train_feature_x = np.array(train_feature_x)
        self.train_output_y = np.array(train_output_y)
        self.initial_weight = np.zeros(self.train_feature_x.shape[1])
        
    def feature_derivative(self, train_feature_x, errors):
        """
        compute derivative
        """
        # Compute the dot product of errors and feature
        derivative = np.dot(train_feature_x.T, errors)
        return derivative
        
    def predict_probability(self, x, weights):
        """
        produces probablistic estimate for P(y_i = +1 | x_i, w).
        estimate ranges between 0 and 1.
        """
        # Take dot product of feature_matrix and coefficients  
        scores = np.dot(x, weights)
    
        # Compute P(y_i = +1 | x_i, w) using the link function
        probs = 1./(1. + np.exp(-scores))
    
        # return probs predictions
        return scores, probs
        
    def compute_log_likelihood(self, train_feature_x,indicators, weights):
        """
        implement log likelihood function to assess the algorithm
        """
        scores, probs = self.predict_probability(train_feature_x, weights)
        logexp = np.log(1./probs)
        # Simple check to prevent overflow
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask] 
        lp = np.sum((indicators-1)*scores - logexp)/len(train_feature_x)
        return lp    
        
    def prediction(self, x, weights):
        """
        predicts based on the scores
        """
        scores, _ = self.predict_probability(x, weights)
        preds = []
        for s in scores:
            if s>0: preds.append(1)
            else: preds.append(-1)
        return preds
        
    def prediction_prob(self, x, weights):
        """
        predicts based on the probability
        """
        _, probs = self.predict_probability(x, weights)
        preds = []
        for p in probs:
            if p>0.5: preds.append(1)
            else: preds.append(-1)
        return preds
        
    def sga_gradient_model (self, x, batch_size, initial_data_start = 0, \
                         initial_weights = None, \
                         step_size = 1.0e-7, tol = 2.5e+7, n_iters = 301):
        """
        This model calculate weights based on gradient ascent solution
        and predict output
        """
        log_likelihood_all = []
        # setup initial intercept, slope, iter number and rss
        if initial_weights is None:
            weights = self.initial_weight
        else:
            weights = initial_weights
        data_start = initial_data_start     
        # set seed=1 to produce consistent results
        np.random.seed(seed=1)
        # Shuffle the data before starting
        permutation = np.random.permutation(len(self.train_feature_x))
        train_feature_x = self.train_feature_x[permutation,:]
        train_output_y = self.train_output_y[permutation]
        # Compute indicator value for (y_i = +1)
        indicators = np.array([int (i) for i in (train_output_y==1)])
        for itr in range(n_iters):
            # Predict P(y_i = +1|x_1,w) using your predict_probability() function
            _, pred_probs = self.predict_probability(train_feature_x\
                                [data_start:data_start+batch_size, :], weights)
            
            # Compute the errors as indicator - predictions
            errors = indicators[data_start:data_start+batch_size] - pred_probs

            #Update the weights:
            derivative = self.feature_derivative(train_feature_x\
                                [data_start:data_start+batch_size, :], errors)
            weights = weights + derivative * (step_size)        
            
            #check if converged
            #todo
            lp = self.compute_log_likelihood(train_feature_x\
                     [data_start:data_start+batch_size, :],\
                     indicators[data_start:data_start+batch_size],weights)
            log_likelihood_all.append(lp)
            # Checking whether log likelihood is increasing
            if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
            or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
                 print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                 (int(np.ceil(np.log10(n_iters))), itr, lp)
                    # if we made a complete pass over data, shuffle and restart
            data_start += batch_size
            if data_start+batch_size > len(self.train_feature_x):
                permutation = np.random.permutation(len(self.train_feature_x))
                train_feature_x = self.train_feature_x[permutation,:]
                train_output_y = self.train_output_y[permutation]
                data_start = initial_data_start  
        #check weights
        #print "\n"
        #print "The weights for features: ", weights
        #final prediction
        preds = self.prediction(x, weights)
        return preds, log_likelihood_all        
                 
if __name__ == "__main__":

    #help functions

    def remove_punctuation(text):
        """
         remove punctuation from text
        """
        return text.translate(None, string.punctuation)

    def get_matrix(df, features, output):
        """
        transform data from pandas dataframe to matrix
        """
        #add a constant column as coefficient for w0
        df["constant"] = 1.0
        feature_x, output_y = df[features].astype(float), df[output].astype(float)
        return feature_x, output_y
    
    def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
        """
        log_likelihood_all, the list of average log likelihood over time
        len_data, number of data points in the training set
        batch_size, size of each mini-batch
        smoothing_window, a parameter for computing moving averages
        """
        plt.figure()
        plt.rcParams.update({'figure.figsize': (9,5)})
        log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

        plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
        plt.rcParams.update({'font.size': 16})
        plt.tight_layout()
        plt.xlabel('# of passes over data')
        plt.ylabel('Average log likelihood per data point')
        plt.legend(loc='lower right', prop={'size':14})
        
    def accuracy_score(preds, y):
        """
        compute accuracy score for predictions
        """
        accuracy = sum([1 for i in range (len(preds)) if preds[i] == y[i]])*1.0/len(preds)   
        return accuracy
        
    #load file
    zf = ZipFile("amazon_baby_subset.csv.zip")
    products = pd.read_csv(zf.open("amazon_baby_subset.csv"))
    #replace NA values with empty string
    products = products.fillna({"review":""})
    #clean review data: remove punctuation
    products["review_clean"] = products["review"].apply(remove_punctuation)

    #load import words
    features = ["constant"]
    with open ("important_words.json", "r") as f:
        important_words = json.load(f)
    for word in important_words:
        features.append(str(word))
        products[str(word)] = products['review_clean'].apply(lambda s : s.split().count(word))
    
    with open ("module-10-assignment-train-idx.json", "r") as f1, \
         open("module-10-assignment-validation-idx.json", "r") as f2:
         train_idx = json.load(f1)
         val_idx = json.load(f2)
    
    train_data = products.iloc[train_idx]
    val_data = products.iloc[val_idx]
    print len(train_data)
    
    #get features matrix and target label
    train_feature_x, train_output_y = get_matrix(train_data, features, "sentiment")
    val_feature_x, val_output_y = get_matrix(val_data, features, "sentiment")  
      
    """
    We now run stochastic gradient ascent over the feature_matrix_train for 10 iterations using:
    initial_coefficients = np.zeros(194)
    step_size = 5e-1
    batch_size = 1
    max_iter = 10
    Quiz Question. When you set batch_size = 1, as each iteration passes, 
    how does the average log likelihood in the batch change?
    """
    step_size = 5e-1
    batch_size = 1
    max_iter = 10
    model = LogisticRegressionSGA(train_feature_x, train_output_y)
    _, log_likelihood_all = model.sga_gradient_model(x = train_feature_x, \
                          step_size = step_size, batch_size = batch_size, n_iters = max_iter)  
    print "The average log likelihood: ", log_likelihood_all, "\n"
    
    #Quiz Question. When you set batch_size = len(train_data), as each 
    #iteration passes, how does the average log likelihood in the batch change?
    _, log_likelihood_all = model.sga_gradient_model(x = train_feature_x, \
                          step_size = step_size, batch_size = len(train_feature_x),\
                          n_iters = 200)  
    print "The average log likelihood: ", log_likelihood_all, "\n"

    _, log_likelihood_all = model.sga_gradient_model(x = train_feature_x, \
                          step_size = 1e-1, batch_size = 100,\
                          n_iters = len(train_feature_x)*10/100)  
    make_plot(log_likelihood_all, len_data = len(train_feature_x), batch_size =100)
    
    
    _, log_likelihood_all = model.sga_gradient_model(x = train_feature_x, \
                          step_size = 1e-1, batch_size = 100,\
                          n_iters = len(train_feature_x)*200/100) 
    make_plot(log_likelihood_all, len_data = len(train_feature_x), batch_size =100,\
               smoothing_window=30)
    _, log_likelihood_all = model.sga_gradient_model(x = train_feature_x, \
                          step_size = 5e-1, batch_size = len(train_feature_x),\
                          n_iters = len(train_feature_x)*200/len(train_feature_x)) 
    make_plot(log_likelihood_all, len_data = len(train_feature_x), \
                   batch_size =len(train_feature_x), smoothing_window=30)
    
    step_sizes = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2]
    for step_size in step_sizes:
        _, log_likelihood_all = model.sga_gradient_model(x = train_feature_x, \
            step_size = step_size, batch_size = 100,\
                          n_iters = len(train_feature_x)*10/100) 
        make_plot(log_likelihood_all, len_data = len(train_feature_x), batch_size =100,\
               smoothing_window=30)