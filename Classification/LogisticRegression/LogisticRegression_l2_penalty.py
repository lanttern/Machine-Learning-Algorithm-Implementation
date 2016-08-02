# -*- coding: utf-8 -*-
"""
Created on Thu March 19 20:20:35 2016

@author: zhihuixie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string, json
from zipfile import ZipFile
    
class LogisticRegression_l2():
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
        
    def feature_derivative(self, errors, weights, l2, is_feature_constant = False):
        """
        compute derivative, regularize constant feature if is_feature_constant is true
        otherwise don't regularize constant feature
        """
        # Compute the dot product of errors and feature
        if is_feature_constant:   
            derivative = np.dot(self.train_feature_x.T, errors) - 2*l2*weights
        else:
            weight0 = np.zeros(1)
            if weights.shape[0] == 2:
                weight1 = np.array(weights[1])
            else: weight1 = weights[1:]
            derivative = np.dot(self.train_feature_x.T, errors) - np.concatenate((weight0, 2*l2*weight1), axis = 0)
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
        
    def compute_log_likelihood(self, indicators, weights, l2):
        """
        implement log likelihood function to assess the algorithm
        """
        scores, _ = self.predict_probability(self.train_feature_x, weights)
        probs = self.predict_probability(self.train_feature_x, weights)
        lp = np.sum((indicators-1)*scores + np.log(probs)) - l2* np.sum(weights[1:]**2)
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
                         step_size = 5.0e-6, tol = 2.5e+7, n_iters = 501, l2 = 0):
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
            derivative = self.feature_derivative(errors, weights, l2)
            weights = weights + derivative * (step_size)        
            
            #check if converged
            #todo
            """
            # Checking whether log likelihood is increasing
            if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
            or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
                 lp = self.compute_log_likelihood(indicators,weights)
                 print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                 (int(np.ceil(np.log10(n_iters))), itr, lp)
            """
        
        #check weights
        #print "\n"
        #print "The weights for features: ", weights
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
        feature_x, output_y = df[features].astype(float), df[output].astype(int)
        return feature_x, output_y
    
    def visualization(positive_weights_dict, negative_weights_dict, l2_list):
        """
        visualize the prediction results
        color negative weights as blue, positive words as red
        """
        plt.figure()
        cmap_positive = plt.get_cmap('Reds')
        cmap_negative = plt.get_cmap('Blues')
        xx = l2_list
        plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
        i, j = 0, 0
        for key, value in positive_weights_dict.items():
            color = cmap_positive(0.8*((i+1)/(5*1.2)+0.15))
            plt.plot(l2_list, value, '-', label=key, linewidth=4.0, color=color)
            i += 1
        for key, value in negative_weights_dict.items():
            color = cmap_negative(0.8*((j+1)/(5*1.2)+0.15))
            plt.plot(l2_list, value, '-', label=key, linewidth=4.0, color=color)
            j += 1
            
        plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
        plt.axis([1, 1e5, -1, 2])
        plt.title('Coefficient path')
        plt.xlabel('L2 penalty ($\lambda$)')
        plt.ylabel('Coefficient value')
        plt.xscale('log')
        plt.rcParams.update({'font.size': 18})
        plt.tight_layout()
        
    def accuracy_score(preds, y):
        """
        compute accuracy score for predictions
        """
        accuracy = sum([1 for i in range (len(preds)) if preds[i] == y[i]])*1.0/len(preds)   
        return accuracy
        
    #load file
    zf = ZipFile("amazon_baby_subset.csv.zip")
    products = pd.read_csv(zf.open("amazon_baby_subset.csv"))
        #get train and validation dataset
    with open ("module-4-assignment-train-idx.json", "r") as f1:
        train_idx = json.load(f1)
        
    with open ("module-4-assignment-validation-idx.json", "r") as f2:
        val_idx = json.load(f2)
    train = products.iloc[train_idx, :]
    val = products.iloc[val_idx, :]

    #replace NA values with empty string
    train = train.fillna({"review":""})
    val = val.fillna({"review":""})
    #clean review data: remove punctuation
    train["review_clean"] = train["review"].apply(remove_punctuation)
    val["review_clean"] = val["review"].apply(remove_punctuation)

    #load import words
    features = ["constant"]
    with open ("important_words.json", "r") as f:
        important_words = json.load(f)
    for word in important_words:
        features.append(str(word))
        train[str(word)] = train['review_clean'].apply(lambda s : s.split().count(word))
        val[str(word)] = val['review_clean'].apply(lambda s : s.split().count(word))
    
    
    #get features matrix and target label
    train_feature_x, train_output_y = get_matrix(train, features, "sentiment")
    val_feature_x, val_output_y = get_matrix(val, features, "sentiment")
    #print train_output_y.values,  type(train_output_y.values), train_output_y == 1
    
    #Questions
    """
    Let us train models with increasing amounts of regularization, starting 
    with no L2 penalty, which is equivalent to our previous logistic regression 
    implementation. Train 6 models with L2 penalty values 0, 4, 10, 1e2, 1e3, 
    and 1e5. Use the following values for the other parameters:

    feature_matrix = feature_matrix_train extracted in #7
    sentiment = sentiment_train extracted in #7
    initial_coefficients = a 194-dimensional vector filled with zeros
    step_size = 5e-6
    max_iter = 501
    Save the 6 sets of coefficients as coefficients_0_penalty, 
    coefficients_4_penalty, coefficients_10_penalty, coefficients_1e2_penalty, 
    coefficients_1e3_penalty, and coefficients_1e5_penalty respectively.
    """
    #Which of the following is not listed in either positive_words or negative_words?
    model = LogisticRegression_l2(train_feature_x, train_output_y)
    _, weights = model.gradient_model(train_feature_x)
    words_weights = zip(features[1:], weights[1:])
    sorted_weights = sorted(words_weights, key = lambda s: s[1])
    positive_words = [word for (word, weight) in sorted_weights[-5:]]
    negative_words = [word for (word, weight) in sorted_weights[:5]]
    print "Positive words: ", positive_words
    print "Negative words: ", negative_words, "\n"
    #Quiz Question: (True/False) All coefficients consistently get smaller in 
    #size as L2 penalty is increased.
    #Quiz Question: (True/False) Relative order of coefficients is preserved as 
    #L2 penalty is increased. (If word 'cat' was more positive than word 'dog', 
    #then it remains to be so as L2 penalty is increased.)
    positive_weights_dict = {}
    negative_weights_dict = {}
    train_accuracy = []
    for word in positive_words:
        positive_weights_dict[word] = []
    for word in negative_words:
        negative_weights_dict[word] = []
    l2_list = [0, 4, 10, 1e2, 1e3, 1e5]
    for l2 in l2_list:
        preds,weights = model.gradient_model(train_feature_x, l2 = l2)
        accuracy = accuracy_score(preds, train_output_y.values)
        train_accuracy.append(accuracy)
        words_weights = zip(features[1:], weights[1:])
        for word_weight in words_weights:
            if word_weight[0] in positive_words:
                positive_weights_dict[word_weight[0]].append(word_weight[1])
            if word_weight[0] in negative_words:
                negative_weights_dict[word_weight[0]].append(word_weight[1])
                
    visualization(positive_weights_dict, negative_weights_dict, l2_list)

    
    #Quiz question: Which model (L2 = 0, 4, 10, 100, 1e3, 1e5) has the highest 
    #accuracy on the training data?
    train_accu_l2 = zip(l2_list, train_accuracy)
    sorted_train_accu_l2 = sorted(train_accu_l2, key = lambda x: x[1])
    print "The highest accuracy on train data: ", sorted_train_accu_l2[-1], "\n"
    
    #Quiz question: Which model (L2 = 0, 4, 10, 100, 1e3, 1e5) has the highest 
    #accuracy on the validation data?
    val_accuracy = []
    for l2 in l2_list:
        preds,weights = model.gradient_model(val_feature_x, l2 = l2)
        accuracy = accuracy_score(preds, val_output_y.values)
        val_accuracy.append(accuracy)
        
    val_accu_l2 = zip(l2_list, val_accuracy)
    sorted_val_accu_l2 = sorted(val_accu_l2, key = lambda x: x[1])
    print "The highest accuracy on validation data: ", sorted_val_accu_l2[-1], "\n"
    
    #Quiz question: Does the highest accuracy on the training data imply that 
    #the model is the best one?
    preds,weights = model.gradient_model(val_feature_x, l2 = sorted_train_accu_l2[-1][0])
    accu = accuracy_score(preds, val_output_y.values)
    
    print "The accuracy on validation data using l2 with highest accuracy on training data: ",\
          accu, "\n"
    
          


