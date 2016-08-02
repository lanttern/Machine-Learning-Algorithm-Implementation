# -*- coding: utf-8 -*-
"""
Created on April 1 20:01:35 2016

@author: zhihuixie
"""
import pandas as pd
from zipfile import ZipFile
import json


class DecisionTrees():
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

    def intermediate_node_num_mistakes(self, labels_in_node):
        """
        calculate number of misclassify
        """
        # Corner case: If labels_in_node is empty, return 0
        if len(labels_in_node) == 0:
            return 0    
        # Count the number of 1's (safe loans)
        count_safe_loans = sum([i for i in labels_in_node if i == 1])
        # Count the number of -1's (risky loans)
        count_risk_loans = sum([-i for i in labels_in_node if i == -1])
        # Return the number of mistakes that the majority classifier makes.
        if count_safe_loans >= count_risk_loans:
            return count_risk_loans
        else: return count_safe_loans
          
    def best_splitting_feature(self, remain_train_data, features):
        """
        select best feature for spliting
        """
        
        best_feature = None # Keep track of the best feature 
        best_error = 10     # Keep track of the best error so far 
        # Note: Since error is always <= 1, we should intialize it with something larger than 1.

        # Convert to float to make sure error gets computed correctly.
        num_data_points = float(remain_train_data.shape[0])  
    
        # Loop through each feature to consider splitting on that feature
        for feature in features:
        
            # The left split will have all data points where the feature value is 0
            left_split = remain_train_data[remain_train_data[feature] == 0]
        
            # The right split will have all data points where the feature value is 1
            right_split =  remain_train_data[remain_train_data[feature] == 0]
            
            # Calculate the number of misclassified examples in the left split.
            # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
            left_mistakes = self.intermediate_node_num_mistakes(left_split[self.label])            

            # Calculate the number of misclassified examples in the right split.
            right_mistakes = self.intermediate_node_num_mistakes(right_split[self.label]) 
            
            # Compute the classification error of this split.
            # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
            error = (left_mistakes + right_mistakes)/num_data_points

            # If this is the best error we have found so far, 
            #store the feature as best_feature and the error as best_error
            if error < best_error:
                best_error = error
                best_feature = feature
        return best_feature # Return the best feature we found
    
    def create_leaf(self,target_values):    
        # Create a leaf node
        leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True }   ## YOUR CODE HERE 
   
        # Count the number of data points that are +1 and -1 in this node.
        num_ones = len(target_values[target_values == +1])
        num_minus_ones = len(target_values[target_values == -1])    

        # For the leaf node, set the prediction to be the majority class.
        # Store the predicted class (1 or -1) in leaf['prediction']
        if num_ones > num_minus_ones:
            leaf['prediction'] = 1 
        else:
            leaf['prediction'] = -1

        # Return the leaf node
        return leaf 
    
    
    def decision_tree_create(self, data, features, current_depth = 0, \
                     max_depth = 10, min_node_size=1, min_error_reduction=0.0):
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
        if self.intermediate_node_num_mistakes(target_values) == 0:
            #print "Stopping condition 1 reached."     
            # If not mistakes at current node, make current node a leaf node
            return self.create_leaf(target_values)
    
        # Stopping condition 2 (check if there are remaining features to consider splitting on)
        if len(remaining_features) == 0:
            #print "Stopping condition 2 reached."    
            # If there are no remaining features to consider, make current node a leaf node
            return self.create_leaf(target_values)    
    
        # Additional stopping condition (limit tree depth)
        if current_depth >= max_depth:  ## YOUR CODE HERE
            #print "Reached maximum depth. Stopping for now."
            # If the max tree depth has been reached, make current node a leaf node
            #print "Early stopping condition 1 reached. Reached minimum node size."
            return self.create_leaf(target_values)
        if len(data) <= min_node_size: 
            #early stopping condition2
            #Early stopping condition 2: Minimum node size
            #print "Early stopping condition 2 reached. Reached minimum node size."
            return self.create_leaf(target_values)
        # Find the best splitting feature (recall the function best_splitting_feature implemented above)
        splitting_feature = self.best_splitting_feature(data, remaining_features)
    
        # Split on the best feature that we found. 
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        left_mistakes = self.intermediate_node_num_mistakes(left_split[self.label])            

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = self.intermediate_node_num_mistakes(right_split[self.label]) 
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error_before_split = self.intermediate_node_num_mistakes(target_values)/float(len(target_values))
        error_after_split = (left_mistakes + right_mistakes)/float(len(target_values)) 
        # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
        if error_after_split - error_before_split <= min_error_reduction:      
            #print "Early stopping condition 3 reached. Minimum error reduction."
            return self.create_leaf(target_values)  
            
        remaining_features = [feature for feature in remaining_features if feature != splitting_feature]
        #print "Split on feature %s. (%s, %s)" % (\
        #                  splitting_feature, len(left_split), len(right_split))
            
            
        # Create a leaf node if the split is "perfect"
        if len(left_split) == len(data):
            #print "Creating leaf node."
            return self.create_leaf(left_split[self.label])
        if len(right_split) == len(data):
            #print "Creating leaf node."
            return self.create_leaf(right_split[self.label])
       
        # Repeat (recurse) on left and right subtrees
        left_tree = self.decision_tree_create(left_split, remaining_features, \
            current_depth + 1, max_depth, min_node_size, min_error_reduction)        
  
        right_tree = self.decision_tree_create(right_split, remaining_features, \
            current_depth + 1, max_depth, min_node_size, min_error_reduction) 
            
        return {'is_leaf'          : False, 
                'prediction'       : None,
                'splitting_feature': splitting_feature,
                'left'             : left_tree, 
                'right'            : right_tree}


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

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])


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
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]

    target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

    # Extract the feature columns and target column
    loans = loans[features + [target]]
    #print loans.head(5)
        
    # one hot encode
    cols_cat = loans.select_dtypes(include = ["object"]).columns
    cols_num = loans.select_dtypes(include = ["int64", "float64"]).columns
    #loans_data_cat = one_encoder(loans, cols_cat)
    loans_data_cat = pd.get_dummies(loans[cols_cat])
    loans_data_num = loans[cols_num]
    
    loans_data = loans_data_cat.join(loans_data_num)
    loans_data = loans_data.fillna(0)
    
    #split train validation dataset
    with open("module-6-assignment-train-idx.json", "r") as f1, \
         open("module-6-assignment-validation-idx.json", "r") as f2:
             train_idx = json.load(f1)
             val_idx = json.load(f2)
    train_data = loans_data.iloc[train_idx]
    val_data = loans_data.iloc[val_idx]
    new_features = train_data.columns.values
    
    my_decision_tree = DecisionTrees(train_data, label = target)
    
    #train model with early stopping        
    new_tree = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 100, min_error_reduction = 0.0)
    #train model without early stopping    
    old_tree = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 0, min_error_reduction = -1)
    
    print val_data.iloc[0]
    """
    Quiz question: For my_decision_tree_new trained with max_depth = 6, 
    min_node_size = 100, min_error_reduction=0.0, is the prediction path for 
    validation_set[0] shorter, longer, or the same as for my_decision_tree_old 
    that ignored the early stopping conditions 2 and 3?

    Quiz question: For my_decision_tree_new trained with max_depth = 6, 
    min_node_size = 100, min_error_reduction=0.0, is the prediction path 
    for any point always shorter, always longer, always the same, shorter or 
    the same, or longer or the same as for my_decision_tree_old that ignored 
    the early stopping conditions 2 and 3?

    Quiz question: For a tree trained on any dataset using max_depth = 6, 
    min_node_size = 100, min_error_reduction=0.0, what is the maximum number of
    splits encountered while making a single prediction?
    """
    print "Predicted class with early stopping: %s " \
       % my_decision_tree.classify(new_tree, val_data.iloc[0], annotate = True), "\n"
    print "Predicted class without early stopping: %s " \
       % my_decision_tree.classify(old_tree, val_data.iloc[0], annotate = True), "\n"
       
    #Quiz question: Is the validation error of the new decision tree 
    #(using early stopping conditions 2 and 3) lower than, higher than, or 
    #the same as that of the old decision tree from the previous assignment?
       
    # Apply the classify(tree, x) to each row in your data
    old_prediction = [my_decision_tree.classify(old_tree, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    y = val_data["safe_loans"].values
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == old_prediction[i]])/len(y)
    print "Classification error for old tree is: %.2f"%(1.0-score), "\n"
    
    # Apply the classify(tree, x) to each row in your data
    new_prediction = [my_decision_tree.classify(new_tree, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == new_prediction[i]])/len(y)
    print "Classification error for new tree is: %.2f"%(1.0-score), "\n"

    """
    Exploring the effect of max_depth

    We will compare three models trained with different values of the stopping 
    criterion. We intentionally picked models at the extreme ends (too small, 
    just right, and too large).

    22. Train three models with these parameters:

    model_1: max_depth = 2 (too small)
    model_2: max_depth = 6 (just right)
    model_3: max_depth = 14 (may be too large)
    For each of these three, set min_node_size = 0 and min_error_reduction = -1. 
    Make sure to call the models model_1, model_2, and model_3.
    Quiz Question: Which tree has the smallest error on the validation data?

    Quiz Question: Does the tree with the smallest error in the training data 
    also have the smallest error in the validation data?

    Quiz Question: Is it always true that the tree with the lowest classification 
    error on the training set will result in the lowest classification error 
    in the validation set?
    """
    tree1 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 2,\
                                 min_node_size = 0, min_error_reduction = -1)
    tree2 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 0, min_error_reduction = -1)
    tree3 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 14,\
                                 min_node_size = 0, min_error_reduction = -1)
                                 
    prediction1 = [my_decision_tree.classify(tree1, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction1[i]])/len(y)
    print "Classification error for model_1 is: %.2f"%(1.0-score), "\n"                             

    prediction2 = [my_decision_tree.classify(tree2, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction2[i]])/len(y)
    print "Classification error for model_2 is: %.2f"%(1.0-score), "\n"  
    
    prediction3 = [my_decision_tree.classify(tree3, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction3[i]])/len(y)
    print "Classification error for model_3 is: %.2f"%(1.0-score), "\n"
    
    """
    Using the function count_leaves, compute the number of nodes in model_1, model_2, and model_3.
    Quiz question: Which tree has the largest complexity?
    Quiz question: Is it always true that the most complex tree will result in 
    the lowest classification error in the validation_set?
    """
    print "The complexity for model_1 %d"%count_leaves(tree1), "\n"
    print "The complexity for model_2 %d"%count_leaves(tree2), "\n"
    print "The complexity for model_3 %d"%count_leaves(tree3), "\n"
    
    
    """
    Exploring the effect of min_error

    We will compare three models trained with different values of the stopping 
    criterion. We intentionally picked models at the extreme ends (negative, 
    just right, and too positive).

    27. Train three models with these parameters:

    model_4: min_error_reduction = -1 (ignoring this early stopping condition)
    model_5: min_error_reduction = 0 (just right)
    model_6: min_error_reduction = 5 (too positive)
    For each of these three, we set max_depth = 6, and min_node_size = 0. 
    Make sure to call the models model_4, model_5, and model_6.
    """
    tree4 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 0, min_error_reduction = -1)
    tree5 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 0, min_error_reduction = 0)
    tree6 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 0, min_error_reduction = 5)
                                 
    """
    Quiz Question: Using the complexity definition above, which model 
    (model_4, model_5, or model_6) has the largest complexity? Did this match 
    your expectation?
    Quiz Question: model_4 and model_5 have similar classification error on the 
     validation set but model_5 has lower complexity? Should you pick model_5 over model_4?
   """
    prediction4 = [my_decision_tree.classify(tree4, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction4[i]])/len(y)
    print "Classification error for model_4 is: %.2f"%(1.0-score), "\n"                             

    prediction5 = [my_decision_tree.classify(tree5, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction5[i]])/len(y)
    print "Classification error for model_5 is: %.2f"%(1.0-score), "\n"  
    
    prediction6 = [my_decision_tree.classify(tree6, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction6[i]])/len(y)
    print "Classification error for model_6 is: %.2f"%(1.0-score), "\n"
    print "The complexity for model_4 %d"%count_leaves(tree4), "\n"
    print "The complexity for model_5 %d"%count_leaves(tree5), "\n"
    print "The complexity for model_6 %d"%count_leaves(tree6), "\n"
    
    """
    Exploring the effect of min_node_size

    We will compare three models trained with different values of the stopping 
    criterion. Again, intentionally picked models at the extreme ends (too small, 
    just right, and just right).

    30. Train three models with these parameters:

    model_7: min_node_size = 0 (too small)
    model_8: min_node_size = 2000 (just right)
    model_9: min_node_size = 50000 (too large)
    For each of these three, we set max_depth = 6, and min_error_reduction = -1. 
    Make sure to call these models model_7, model_8, and model_9.
    """
    tree7 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 0, min_error_reduction = -1)
    tree8 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 2000, min_error_reduction = -1)
    tree9 = my_decision_tree.decision_tree_create(data = train_data, \
                                 features = new_features[:-1], max_depth = 6,\
                                 min_node_size = 50000, min_error_reduction = -1)
                                 
    """
    31. Calculate the accuracy of each model (model_7, model_8, or model_9) on 
    the validation set.

    32. Using the count_leaves function, compute the number of leaves in each 
    of each models (model_7, model_8, and model_9).

    Quiz Question: Using the results obtained in this section, which model 
    (model_7, model_8, or model_9) would you choose to use?
    """
    prediction7 = [my_decision_tree.classify(tree7, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction7[i]])/len(y)
    print "Classification error for model_7 is: %.2f"%(1.0-score), "\n"                             

    prediction8 = [my_decision_tree.classify(tree8, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction8[i]])/len(y)
    print "Classification error for model_8 is: %.2f"%(1.0-score), "\n"  
    
    prediction9 = [my_decision_tree.classify(tree9, val_data.iloc[i]) for i in range(val_data.shape[0])]
    # Once you've made the predictions, calculate the classification error and return it
    score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction9[i]])/len(y)
    print "Classification error for model_9 is: %.2f"%(1.0-score), "\n"
    print "The complexity for model_7 %d"%count_leaves(tree7), "\n"
    print "The complexity for model_8 %d"%count_leaves(tree8), "\n"
    print "The complexity for model_9 %d"%count_leaves(tree9), "\n"
    