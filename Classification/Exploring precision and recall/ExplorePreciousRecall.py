# -*- coding: utf-8 -*-
"""
Created on Fri April 15 19:44:27 2016

@author: zhihuixie
"""

from zipfile import ZipFile
import math
import numpy as np
import pandas as pd
import string, json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

def remove_punctuation(text):
    """
    remove punctuation from text
    """
    return text.translate(None, string.punctuation)

def apply_threshold(probs, thred):
    """
    The function should return an array, where each element is set to +1 or -1 
    depending whether the corresponding probability exceeds threshold.
    """
    preds = [1 if p >= thred else -1 for p in probs]
    return np.array(preds)

def plot_pr_curve(precision, recall, title):
    plt.figure()
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})


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
    with open ("module-9-assignment-train-idx.json", "r") as f1:
        train_index = json.load(f1)
    with open ("module-9-assignment-test-idx.json", "r") as f2:
        test_index = json.load(f2)
    f1.close()
    f2.close()
    #retrivew train and test data 
    train_data = products.iloc[train_index]
    test_data = products.iloc[test_index]

    #compute the word count for each word that appears in the reviews
    #Use this token pattern to keep single-letter words)
    vectorizer = CountVectorizer(token_pattern=r"\b\w+\b")
    train_matrix = vectorizer.fit_transform(train_data["review_clean"])
    test_matrix = vectorizer.transform(test_data["review_clean"])
    
    #train LogisticRegression model
    train_y = train_data["sentiment"]
    model = LogisticRegression()
    model.fit(train_matrix, train_y)
    test_pred = model.predict(test_matrix)
    test_y = test_data["sentiment"]
    #calculate accuracy
    accuracy_model = accuracy_score(test_y, test_pred)
    accuracy_base = len(test_data[test_data["sentiment"] == 1])*1.0/len(test_data)
    
    #Quiz question: Using accuracy as the evaluation metric, was our logistic 
    #regression model better than the baseline (majority class classifier)?
    print "The accuracy for model: %.4f and for baseline: %.4f."%(accuracy_model,\
                                                         accuracy_base), "\n"
    #confusion matrix
    cmat = confusion_matrix(y_true=test_y,y_pred=test_pred,labels=model.classes_)   
    # use the same order of class as the LR model.
    print ' target_label | predicted_label | count '
    print '--------------+-----------------+-------'
    # Print out the confusion matrix.
    # NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
    for i, target_label in enumerate(model.classes_):
        for j, predicted_label in enumerate(model.classes_):
            print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, \
                                                   predicted_label, cmat[i,j])
    #Quiz Question: How many predicted values in the test set are false positives?
    print "\n"
    
    #Computing the cost of mistakes
    true_neg = cmat[0,0]
    false_pos = cmat[0,1]
    false_neg = cmat[1,0]
    true_pos = cmat[1,1]
    
    """
    #Suppose you know the costs involved in each kind of mistake:

    #$100 for each false positive.
    #$1 for each false negative.
    #Correctly classified reviews incur no cost.
    #Quiz Question: Given the stipulation, what is the cost associated with the 
    #logistic regression classifier's performance on the test set?
    """
    cost = false_pos*100 + false_neg*1
    print "The cost for prediction on test data: %d"%(cost), "\n"

    #Precision and Recall
    precision = precision_score(test_y, test_pred)
    #Quiz Question: Out of all reviews in the test set that are predicted to be 
    #positive, what fraction of them are false positives? (Round to the second 
    #decimal place e.g. 0.25)
    print "False positive rate on test data: %.4f" %(1.0 - precision), "\n"
    #Quiz Question: Based on what we learned in lecture, if we wanted to reduce 
    #this fraction of false positives to be below 3.5%, we would: (see quiz)
    recall = recall_score(test_y, test_pred)
    #Quiz Question: What fraction of the positive reviews in the test_set 
    #were correctly predicted as positive by the classifier?
    #Quiz Question: What is the recall value for a classifier that predicts +1 for 
    #all data points in the test_data?
    print "Recall on test data: %.4f" % recall, "\n"
    
    
    #Precision-recall tradeoff
    probs = model.predict_proba(test_matrix)[:,1]
    #Quiz question: What happens to the number of positive predicted reviews as 
    #the threshold increased from 0.5 to 0.9?
    preds_09 = apply_threshold(probs, thred = 0.9) 
    pred_pos09 = len(preds_09[preds_09 == 1]) 
    print "Postive preds for thred as 0.5: %d, for thred as 0.9: %d"%(true_pos+false_pos,pred_pos09), "\n"
    
    #Quiz Question (variant 1): Does the precision increase with a higher threshold?
    #Quiz Question (variant 2): Does the recall increase with a higher threshold?
    precision_09 = precision_score(test_y, preds_09)
    recall_09 = recall_score(test_y, preds_09)
    print "Precision for thred as 0.5: %.4f, for thred as 0.9: %.4f"%(precision,precision_09), "\n"
    print "recall for thred as 0.5: %.4f, for thred as 0.9: %.4f"%(recall,recall_09), "\n"    

    #Precision-recall curve
    threshold_values = np.linspace(0.5, 1, num=100)
    precision_all, recall_all = [], []
    for thred in threshold_values:
        preds = apply_threshold(probs, thred)
        precision_all.append(precision_score(test_y, preds))
        recall_all.append(recall_score(test_y, preds))
        
    plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
    
    #Quiz Question: Among all the threshold values tried, what is the smallest 
    #threshold value that achieves a precision of 0.965 or better? Round your 
    #answer to 3 decimal places.
    p_t = zip(precision_all, threshold_values)
    min_thred = min([t for (p, t) in p_t if p >= 0.965])
    print "The smallest thred to achieve precision of 0.965: %.3f" %(min_thred), "\n"

    #Quiz Question: Using threshold = 0.98, how many false negatives do we get
    # on the test_data? This is the number of false negatives (i.e the number 
    #of reviews to look at when not needed) that we have to deal with using this 
    #classifier.
    preds = apply_threshold(probs, thred = 0.98) 
    y = test_y.values
    false_neg = sum([1 for i in range(len(y)) if y[i] == 1 and preds[i] == -1])
    print "The false negative for thred as 0.98: %d" %(false_neg)
    
    #Evaluating specific search terms
    test_data["name"].fillna("", inplace = True)
    baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]
    test_y = baby_reviews["sentiment"]
    baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
    probs = model.predict_proba(baby_matrix)[:,1]

    precision_all, recall_all = [], []
    for thred in threshold_values:
        preds = apply_threshold(probs, thred)
        precision_all.append(precision_score(test_y, preds))
        recall_all.append(recall_score(test_y, preds))
        
    plot_pr_curve(precision_all, recall_all, "Precision-Recall (Baby)")
    
    #Quiz Question: Among all the threshold values tried, what is the smallest 
    #threshold value that achieves a precision of 0.965 or better? Round your 
    #answer to 3 decimal places.
    p_t = zip(precision_all, threshold_values)
    min_thred = min([t for (p, t) in p_t if p >= 0.965])
    print "The smallest thred to achieve precision of 0.965 for baby review: %.3f" %(min_thred), "\n"
    


