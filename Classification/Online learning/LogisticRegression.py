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
    
class LogisticRegression():
    """
    This class implement logistic regression algorithm. 
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
        compute derivative
        """
        # Compute the dot product of errors and feature
        derivative = np.dot(self.train_feature_x.T, errors)
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
        
    def compute_log_likelihood(self, indicators, weights):
        """
        implement log likelihood function to assess the algorithm
        """
        scores, _ = self.predict_probability(self.train_feature_x, weights)
        probs = self.predict_probability(self.train_feature_x, weights)
        lp = np.sum((indicators-1)*scores + np.log(probs))
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
        
    def gradient_model (self, x, initial_weights = None, \
                         step_size = 1.0e-7, tol = 2.5e+7, n_iters = 301):
        """
        This model calculate weights based on gradient ascent solution
        and predict output
        """
        # setup initial intercept, slope, iter number and rss
        if initial_weights is None:
            weights = self.initial_weight
        else:
            weights = initial_weights
        # Compute indicator value for (y_i = +1)
        indicators = np.array([int (i) for i in (self.train_output_y==1)])
        for itr in range(n_iters):
            # Predict P(y_i = +1|x_1,w) using your predict_probability() function
            _, pred_probs = self.predict_probability(self.train_feature_x, weights)
            
            # Compute the errors as indicator - predictions
            errors = indicators - pred_probs

            #Update the weights:
            derivative = self.feature_derivative(errors)
            weights = weights + derivative * (step_size)        
            
            #check if converged
            #todo
            
            # Checking whether log likelihood is increasing
            if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
            or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
                 lp = self.compute_log_likelihood(indicators,weights)
                 print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                 (int(np.ceil(np.log10(n_iters))), itr, lp)
        
        #check weights
        print "\n"
        print "The weights for features: ", weights
        #final prediction
        preds = self.prediction(x, weights)
        return preds, weights        
                 
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
    
    def visualization(x, y, preds):
        """
        visualize the prediction results
        color real output as blue, prediction as red
        """
        plt.figure()
        plt.plot(x, y, ".", color = "blue")
        plt.plot(x, preds, ".", color = "red")
        
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
    
    #get features matrix and target label
    train_feature_x, train_output_y = get_matrix(products, features, "sentiment")
      
    #QUestions
    #Q2: How many reviews in amazon_baby_subset.gl contain the word perfect?
    reviews = products['review_clean'].values
    perfect_counts = sum([1 for review in reviews if "perfect" in review])
    print "Number of reviews contain 'perfect' :", perfect_counts
    
    #Q3: How many features are there in the feature_matrix?
    print "Number of features in feature_matrix: ", train_feature_x.shape[1]
    
    #Q6: How many reviews were predicted to have positive sentiment?
    model = LogisticRegression(train_feature_x, train_output_y)
    preds, weights = model.gradient_model(train_feature_x)
    num_of_positive_reviews = sum([p for p in preds if p > 0])
    print "Number of positive reviews: ", num_of_positive_reviews
    
    #Q7: What is the accuracy of the model on predictions made above? (round to 2 digits of accuracy)
    accuracy = accuracy_score(preds, train_output_y.values)
    print "The accuracy of the model is: %.2f"%accuracy
    
    #Q8: Which of the following words is not present in the top 10 "most positive" words?
    words_weights = zip(features[1:], weights[1:])
    sorted_weights = sorted(words_weights, key = lambda s: s[1])
    print "The 10 most positive words: ", sorted_weights[-10:] 
    
    #Q9: Which of the following words is not present in the top 10 "most negative" words?
    print "The 10 most negative words: ", sorted_weights[:10] 


