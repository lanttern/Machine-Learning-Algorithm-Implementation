# -*- coding: utf-8 -*-
"""
Created on April 7 20:01:35 2016

@author: zhihuixie
"""
import pandas as pd
from zipfile import ZipFile
import json
import matplotlib.pyplot as plt
from math import log
from math import exp
import numpy as np


class BoostDecisionTree():
    """
    implement greedy algorithm for binary decision tree
    """
    def __init__(self, train_data, label):
        """
        initiate training data as pandas datafram
        and target as a string of label
        """
        self.train_data = train_data
        self.label = label

    def intermediate_node_weighted_mistakes(self, labels_in_node, data_weights):
        """
        calculate weighted number of misclassify
        """
        # Sum the weights of all entries with label +1
        total_weight_positive = sum(data_weights[labels_in_node == +1])
    
        # Weight of mistakes for predicting all -1's is equal to the sum above
        ### YOUR CODE HERE
        weighted_mistakes_all_negative = total_weight_positive
    
        # Sum the weights of all entries with label -1
        ### YOUR CODE HERE
        total_weight_negative = sum(data_weights[labels_in_node == -1])
    
        # Weight of mistakes for predicting all +1's is equal to the sum above
        ### YOUR CODE HERE
        weighted_mistakes_all_positive = total_weight_negative
    
        # Return the tuple (weight, class_label) representing the lower of the two weights
        #    class_label should be an integer of value +1 or -1.
        # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
        ### YOUR CODE HERE
        if weighted_mistakes_all_positive <= weighted_mistakes_all_negative:
            return (weighted_mistakes_all_positive/sum(data_weights),+1)
        else: return (weighted_mistakes_all_negative/sum(data_weights),-1)
          
    def best_splitting_feature(self, remain_train_data, features, data_weights):
        """
        select best feature for spliting
        """
        
        best_feature = None # Keep track of the best feature 
        best_error = float("inf")     # Keep track of the best error so far 
        # Note: Since error is always <= 1, we should intialize it with something larger than 1.

        # Convert to float to make sure error gets computed correctly.
        #num_data_points = float(remain_train_data.shape[0])  
    
        # Loop through each feature to consider splitting on that feature
        for feature in features:
        
            # The left split will have all data points where the feature value is 0
            left_split = remain_train_data[remain_train_data[feature] == 0]
        
            # The right split will have all data points where the feature value is 1
            right_split =  remain_train_data[remain_train_data[feature] == 1]
            left_data_weights = data_weights[remain_train_data[feature] == 0]
            right_data_weights = data_weights[remain_train_data[feature] == 1]
            
            # Calculate the number of misclassified examples in the left split.
            # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
            left_data_weights_mistakes, left_class = self.intermediate_node_weighted_mistakes\
                                                      (left_split[self.label], left_data_weights)            

            # Calculate the number of misclassified examples in the right split.
            right_data_weights_mistakes, right_class = self.intermediate_node_weighted_mistakes\
                                                       (right_split[self.label], right_data_weights)             
            
            # Compute the classification error of this split.
            # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
            error = left_data_weights_mistakes + right_data_weights_mistakes

            # If this is the best error we have found so far, 
            #store the feature as best_feature and the error as best_error
            if error < best_error:
                best_error = error
                best_feature = feature
        return best_feature # Return the best feature we found
    
    def create_leaf(self,target_values, data_weights):    
        # Create a leaf node
        leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True }   ## YOUR CODE HERE 
   
        # Count the number of data points that are +1 and -1 in this node.
        weighted_error, best_class = self.intermediate_node_weighted_mistakes\
                                     (target_values, data_weights)  

        leaf['prediction'] = best_class

        # Return the leaf node
        return leaf 

    def decision_tree_create(self, data, features, data_weights, current_depth = 0, max_depth = 10):
        """
        Stopping condition 1: All data points in a node are from the same class.
        Stopping condition 2: No more features to split on.
        Additional stopping condition: In addition to the above two stopping 
        #conditions covered in lecture, in this assignment we will also consider 
        #a stopping condition based on the max_depth of the tree. By not letting 
        #the tree grow too deep, we will save computational effort in the learning process.   
        """
        
        remaining_features = features[:] # Make a copy of the features.
    
        target_values = data[self.label]
        #print "--------------------------------------------------------------------"
        #print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

        # Stopping condition 1
        # (Check if there are mistakes at current node.)
        if self.intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
            #print "Stopping condition 1 reached."     
            # If not mistakes at current node, make current node a leaf node
            return self.create_leaf(target_values, data_weights)
    
        # Stopping condition 2 (check if there are remaining features to consider splitting on)
        if len(remaining_features) == 0:
            #print "Stopping condition 2 reached."    
            # If there are no remaining features to consider, make current node a leaf node
            return self.create_leaf(target_values, data_weights)    
    
        # Additional stopping condition (limit tree depth)
        if current_depth >= max_depth:  ## YOUR CODE HERE
            #print "Reached maximum depth. Stopping for now."
            # If the max tree depth has been reached, make current node a leaf node
            return self.create_leaf(target_values, data_weights)

        # Find the best splitting feature (recall the function best_splitting_feature implemented above)
        splitting_feature = self.best_splitting_feature(data, remaining_features, data_weights)
    
        # Split on the best feature that we found. 
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]
        left_data_weights = data_weights[data[splitting_feature] == 0]
        right_data_weights = data_weights[data[splitting_feature] == 1]
        remaining_features = [feature for feature in remaining_features if feature != splitting_feature]
        #print "Split on feature %s. (%s, %s)" % (\
        #                  splitting_feature, len(left_split), len(right_split))
    
        # Create a leaf node if the split is "perfect"
        if len(left_split) == len(data):
            #print "Creating leaf node."
            return self.create_leaf(left_split[target], data_weights)
        if len(right_split) == len(data):
            #print "Creating leaf node."
            return self.create_leaf(right_split[target], data_weights)

        
        # Repeat (recurse) on left and right subtrees
        left_tree = self.decision_tree_create(left_split, remaining_features, left_data_weights, current_depth + 1, max_depth)        
        ## YOUR CODE HERE
        right_tree = self.decision_tree_create(right_split, remaining_features,right_data_weights, current_depth + 1, max_depth)
        return {'is_leaf'          : False, 
                'prediction'       : None,
                'splitting_feature': splitting_feature,
                'left'             : left_tree, 
                'right'            : right_tree}

    def adaboost_with_tree_stumps(self, max_depth, num_tree_stumps):
        # start with unweighted data
        data = self.train_data
        features = self.train_data.columns
        alpha = np.array([1.]*len(data))
        weights = []
        tree_stumps = []
        target_values = data[self.label]
    
        for t in xrange(num_tree_stumps):
            print '====================================================='
            print 'Adaboost Iteration %d' % t
            print '====================================================='        
            # Learn a weighted decision tree stump. Use max_depth=1
            print alpha
            tree_stump = self.decision_tree_create(data, features, \
                         data_weights=alpha, max_depth=max_depth)
            tree_stumps.append(tree_stump)
        
            # Make predictions
            predictions = np.array([self.classify(tree_stump, data.iloc[i]) for i in range(len(data))])
        
            # Produce a Boolean array indicating whether
            # each data point was correctly classified
            is_correct = predictions == target_values
            is_wrong = predictions != target_values
        
            # Compute weighted error
            # YOUR CODE HERE
            mis_clarify = is_wrong.apply(lambda is_wrong: 1 if is_wrong else 0)
            weighted_error = np.dot(mis_clarify, alpha)/sum(alpha)
        
            # Compute model coefficient using weighted error
            # YOUR CODE HERE
            print weighted_error
            weight = 1.0/2*log((1-weighted_error)/weighted_error)
            weights.append(weight)
        
            # Adjust weights on data point
            adjustment = np.array(is_correct.apply(lambda is_correct : exp(-weight) if is_correct else exp(weight)))
            print adjustment
            # Scale alpha by multiplying by adjustment
            # Then normalize data points weights
            ## YOUR CODE HERE 
            alpha = alpha*adjustment/sum(alpha)
    
        return weights, tree_stumps


    def classify(self, tree, x, annotate = False):
       # if the node is a leaf node.
        if tree['is_leaf']:
            if annotate:
                 print "At leaf, predicting %s" % tree['prediction']
            return tree['prediction']
        else:
            # split on feature.
            split_feature_value = x[tree['splitting_feature']]
            if annotate:
                 print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
            if split_feature_value == 0:
                return self.classify(tree['left'], x, annotate)
            else:
               return self.classify(tree['right'], x, annotate)
               
               
    def predict_adaboost(self, data, max_depth, num_tree_stumps):
        stump_weights, tree_stumps = self.adaboost_with_tree_stumps(max_depth, num_tree_stumps)       
        scores = np.array([0.]*len(data))
        print stump_weights
        j = 0
        for tree_stump in tree_stumps:
            predictions = np.array([self.classify(tree_stump, data.iloc[i]) for i in range(len(data))])
        
            # Accumulate predictions on scores array
            # YOUR CODE HERE
            scores += predictions*stump_weights[j]
            j += 1
        
        return [1 if score > 0 else -1 for score in scores]



def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


def evaluate_classification_error(y, preds):
 
    # Once you've made the predictions, calculate the classification error
    return (y != preds).sum() / float(len(y))

def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


if __name__ == "__main__":
    #load data
    zf = ZipFile("lending-club-data.csv.zip")
    loans = pd.read_csv(zf.open("lending-club-data.csv"))
    # safe_loans =  1 => safe
    # safe_loans = -1 => risky
    loans["safe_loans"] = loans["bad_loans"].apply(lambda x: 1 if x == 0 else -1)
    del loans["bad_loans"]

    #subset features
    features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]

    target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

    # Extract the feature columns and target column and remove missing data point
    loans = loans[features + [target]]
    # one hot encode
    cols_cat = loans.select_dtypes(include = ["object"]).columns
    cols_num = loans.select_dtypes(include = ["int64", "float64"]).columns
    #loans_data_cat = one_encoder(loans, cols_cat)
    loans_data_cat = pd.get_dummies(loans[cols_cat])
    loans_data_num = loans[cols_num]
    
    loans_data = loans_data_cat.join(loans_data_num)
    new_features = loans_data.columns
    
    #split train validation dataset
    with open("module-8-assignment-2-train-idx.json", "r") as f1, \
         open("module-8-assignment-2-test-idx.json", "r") as f2:
             train_idx = json.load(f1)
             val_idx = json.load(f2)
    train_data = loans_data.iloc[train_idx]
    val_data = loans_data.iloc[val_idx]
    
    example_data_weights = np.array([1.] * 10 + [0.]*(len(train_data) - 20) + [1.] * 10)
    # Train a weighted decision tree model.
    model = BoostDecisionTree(train_data, target)
    
    small_data_decision_tree_subset_20 = model.decision_tree_create\
                    (train_data, new_features[:-1],example_data_weights, max_depth=10)
    """    
    y = train_data[target].values
    preds = np.array([model.classify(small_data_decision_tree_subset_20, train_data.iloc[i]) for i in range(len(train_data))])    
    print evaluate_classification_error(y, preds)
    """
    subset_20 = train_data.iloc[:10].append(train_data.iloc[-10:])
    y = subset_20[target].values
    preds = np.array([model.classify(small_data_decision_tree_subset_20, subset_20.iloc[i]) for i in range(len(subset_20))])     
    print evaluate_classification_error(y, preds)
    
    #model.predict_adaboost(val_data, max_depth = 2, num_tree_stumps = 10)
    