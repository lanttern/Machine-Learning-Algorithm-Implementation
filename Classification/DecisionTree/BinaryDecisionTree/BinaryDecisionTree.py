# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:01:35 2016

@author: zhihuixie
"""
import pandas as pd
from zipfile import ZipFile
import json


class BinaryDecisionTree():
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
            right_split =  remain_train_data[remain_train_data[feature] == 1]
            
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

    def decision_tree_create(self, data, features, current_depth = 0, max_depth = 10):
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
            return self.create_leaf(target_values)

        # Find the best splitting feature (recall the function best_splitting_feature implemented above)
        splitting_feature = self.best_splitting_feature(data, remaining_features)
    
        # Split on the best feature that we found. 
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]
        remaining_features = [feature for feature in remaining_features if feature != splitting_feature]
        #print "Split on feature %s. (%s, %s)" % (\
        #                  splitting_feature, len(left_split), len(right_split))
    
        # Create a leaf node if the split is "perfect"
        if len(left_split) == len(data):
            #print "Creating leaf node."
            return self.create_leaf(left_split[target])
        if len(right_split) == len(data):
            #print "Creating leaf node."
            return self.create_leaf(right_split[target])

        
        # Repeat (recurse) on left and right subtrees
        left_tree = self.decision_tree_create(left_split, remaining_features, current_depth + 1, max_depth)        
        ## YOUR CODE HERE
        right_tree = self.decision_tree_create(right_split, remaining_features, current_depth + 1, max_depth)
        return {'is_leaf'          : False, 
                'prediction'       : None,
                'splitting_feature': splitting_feature,
                'left'             : left_tree, 
                'right'            : right_tree}


    def classify(self, tree, x, annotate = False):
       # if the node is a leaf node.
        print tree
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

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    #split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))




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
    with open("module-5-assignment-2-train-idx.json", "r") as f1, \
         open("module-5-assignment-2-test-idx.json", "r") as f2:
             train_idx = json.load(f1)
             val_idx = json.load(f2)
    train_data = loans_data.iloc[train_idx]
    val_data = loans_data.iloc[val_idx]
    new_features = train_data.columns.values
    
    #train model
    model = BinaryDecisionTree(train_data, label = target)
    
    tree = model.decision_tree_create(data = train_data, features = new_features[:-1])
    
    #print val_data.iloc[0]
    #print 'Predicted class: %s ' % model.classify(tree, val_data.iloc[0])

    #Quiz question: What was the feature that my_decision_tree first split on 
    #while making the prediction for test_data[0]?
    #Quiz question: What was the first feature that lead to a right split of test_data[0]?
    #Quiz question: What was the last feature split on before reaching a leaf node for test_data[0]?
    #print val_data.iloc[0]
    print 'Predicted class: %s ' % model.classify(tree, val_data.iloc[0])


    #Quiz Question: Rounded to 2nd decimal point, what is the classification 
    #error of my_decision_tree on the test_data?

    # Apply the classify(tree, x) to each row in your data
    #prediction = [model.classify(tree, val_data.iloc[i]) for i in range(val_data.shape[0])]
    
    # Once you've made the predictions, calculate the classification error and return it
    #y = val_data["safe_loans"].values
    
    #score = 1.0*sum([1 for i in range(len(y)) if y[i] == prediction[i]])/len(y)
    #print "Classification error is: %.2f"%(1.0-score), "\n"

    #Quiz Question: What is the feature that is used for the split at the root node?
    #print_stump(tree, name = 'root')

    #Quiz question: What is the path of the first 3 feature splits considered 
    #along the left-most branch of my_decision_tree?
    #print_stump(tree['left'], tree['splitting_feature'])

    #Quiz question: What is the path of the first 3 feature splits considered 
    #along the right-most branch of my_decision_tree?
    #print_stump(tree['right']['right']['right'], tree['right']['right']['splitting_feature'])


