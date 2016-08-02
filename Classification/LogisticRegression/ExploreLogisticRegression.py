# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:44:27 2016

@author: zhihuixie
"""

from zipfile import ZipFile
import math
import numpy as np
import pandas as pd
import string, json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

def remove_punctuation(text):
    """
    remove punctuation from text
    """
    return text.translate(None, string.punctuation)

def prediction(scores):
    """
    predict based on score: pred = 1 if score equal to or greater than 0
    pred = -1 when score lower than 0
    """
    preds = []
    for score in scores:
        if score >= 0:
            preds.append(1)
        else:
            preds.append(-1)
    return preds
    
def prob_prediction(scores):
    """
    compute the probability for each score based on sigmoid function
    """
    probs = []
    for score in scores:
        prob = 1.0/(1.0 + math.exp(-score))  
        probs.append(prob)
    return probs







if __name__ == "__main__":
    
    #load file
    zf = ZipFile("amazon_baby.csv.zip")
    products = pd.read_csv(zf.open("amazon_baby.csv"))
    #replace NA values with empty string
    products = products.fillna({"review":""})
    #clean review data: remove punctuation
    products["review_clean"] = products["review"].apply(remove_punctuation)
    #remove neutral rating 3
    products = products[products["rating"] != 3]
    #assign reviews with a rating of 4 or higher to be positive reviews, 
    #while the ones with rating of 2 or lower are negative. 
    products["sentiment"] = products["rating"].apply(lambda x: 1 if x > 3 else -1)
    
    #load train and test data index
    with open ("module-2-assignment-train-idx.json", "r") as f1:
        train_index = json.load(f1)
    with open ("module-2-assignment-test-idx.json", "r") as f2:
        test_index = json.load(f2)
    f1.close()
    f2.close()
    #retrivew train and test data 
    train_data = products.iloc[train_index, :]
    test_data = products.iloc[test_index, :]

    #compute the word count for each word that appears in the reviews
    #Use this token pattern to keep single-letter words)
    vectorizer = CountVectorizer(token_pattern=r"\b\w+\b")
    train_matrix = vectorizer.fit_transform(train_data["review_clean"])
    test_matrix = vectorizer.transform(test_data["review_clean"])
    
    #train LogisticRegression model
    train_y = train_data["sentiment"]
    model = LogisticRegression()
    model.fit(train_matrix, train_y)
    coefficients = model.coef_
    #Quiz question: How many weights are >= 0?
    num = len([weight for weight in coefficients[0] if weight >= 0])
    print "Number of positive weights: %d"%num, "\n"
    
    #test prediction for 3 sample dataset
    sample_test_data = test_data[10:13]
    sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
    scores = model.decision_function(sample_test_matrix)
    #Quiz question: Of the three data points in sample_test_data, which one 
    #(first, second, or third) has the lowest probability of being classified 
    #as a positive review?
    probs = prob_prediction(scores)
    print "The probabilities: ", probs, "\n"
    
    #find the 20 reviews in the entire test_data with the highest probability 
    #of being classified as a positive review.
    scores = model.decision_function(test_matrix)
    probs = prob_prediction(scores)
    index_probs = zip(test_index, probs)
    index_probs.sort(key = lambda x: x[1])
    most_positive_reviews_index = [a for (a, b) in index_probs[-20:]]
    most_positive_reviews = products.iloc[most_positive_reviews_index, 0]
    print "Most positive reviews: ", most_positive_reviews, "\n"
    most_negative_reviews_index = [a for (a, b) in index_probs[:20]]
    most_negative_reviews = products.iloc[most_negative_reviews_index, 0]
    print "Most negative reviews: ", most_negative_reviews, "\n"
    
    #Quiz Question: What is the accuracy of the sentiment_model on the test_data? 
    preds = model.predict(test_matrix)
    test_y = test_data["sentiment"].values
    num_of_accu = sum([1 for i in range(len(preds)) if preds[i] == test_y[i]])
    accu = num_of_accu*1.0/len(test_y)
    print "The accuracy of the model on the test data is: ", accu, "\n"
    
    #Learn another classifier with fewer words
    significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
    vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
    train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
    test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])
    simple_model = LogisticRegression()
    simple_model.fit(train_matrix_word_subset, train_y)
    simple_coefs = simple_model.coef_
    print "The positive coefs: ", len([coef for coef in simple_coefs[0] if coef >= 0]), "\n"
    print "The words and coefs: ", zip(significant_words, simple_coefs[0]), "\n"
    
    #Comparing models
    simple_preds = simple_model.predict(test_matrix_word_subset)
    num_of_accu = sum([1 for i in range(len(simple_preds)) if simple_preds[i] == test_y[i]])
    accu = num_of_accu*1.0/len(test_y)
    print "The accuracy of the simple model on the test data is: ", accu, "\n"
    
    
    # accuracy on train data
    train_y_p = train_data["sentiment"].values
    preds_train = model.predict(train_matrix)
    num_of_accu = sum([1 for i in range(len(preds_train)) if preds_train[i] == train_y_p[i]])
    accu = num_of_accu*1.0/len(train_y_p)
    print "The accuracy of the model on the train data is: ", accu, "\n"
    simple_preds_train = simple_model.predict(train_matrix_word_subset)
    num_of_accu = sum([1 for i in range(len(simple_preds_train)) if simple_preds_train[i] == train_y_p[i]])
    accu = num_of_accu*1.0/len(train_y_p)
    print "The accuracy of the simple model on the train data is: ", accu, "\n"
    
    
    #Baseline: Majority class prediction
    maj_model = DummyClassifier()
    maj_model.fit(train_matrix, train_y)
    preds = maj_model.predict(test_matrix)
    num_of_accu = sum([1 for i in range(len(preds)) if preds[i] == test_y[i]])
    accu = num_of_accu*1.0/len(test_y)
    print "The accuracy of the majority model on the test data is: ", accu, "\n"

    
     