# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:01:35 2016

@author: zhihuixie
"""
import pandas as pd
import numpy as np
from zipfile import ZipFile
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import tree
from subprocess import check_call
def one_encoder(df, cols):
    """
    """
    df = df[cols].fillna(0)
    lbl = LabelEncoder()
    encode_data = []
    for col in cols:
       encode = list(lbl.fit_transform(df[col]))
       encode_data.append(encode)
    encode_df = pd.DataFrame(np.array(encode_data).T)
    encode_df.columns = cols
    encode_df.index = df.index
    return encode_df




if __name__ == "__main__":
    #load data
    zf = ZipFile("lending-club-data.csv.zip")
    loans = pd.read_csv(zf.open("lending-club-data.csv"))
    #Exploring features
    print loans.columns.values
    #exploring target
    print loans["bad_loans"].head(5)
    # safe_loans =  1 => safe
    # safe_loans = -1 => risky
    loans["safe_loans"] = loans["bad_loans"].apply(lambda x: 1 if x == 0 else -1)
    del loans["bad_loans"]
    print loans["safe_loans"].head(5)
    safe_count = sum([1 for i in loans["safe_loans"] == 1 if i])
    risk_count = loans.shape[0] - safe_count
    print safe_count*1./loans.shape[0]
    print risk_count*1./loans.shape[0]

    #subset features
    features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

    target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

    # Extract the feature columns and target column
    loans = loans[features + [target]]
    #print loans.head(5)
    
    with open("module-5-assignment-1-train-idx.json", "r") as f1, \
         open("module-5-assignment-1-validation-idx.json", "r") as f2:
             train_idx = json.load(f1)
             val_idx = json.load(f2)
    train_data = loans.iloc[train_idx]
    val_data = loans.iloc[val_idx]
    
    #sample data to balance class
    safe_loans_raw = loans[loans[target] == +1]
    risky_loans_raw = loans[loans[target] == -1]
    print "Number of safe loans  : %s" % len(safe_loans_raw)
    print "Number of risky loans : %s" % len(risky_loans_raw)
    # Since there are fewer risky loans than safe loans, find the ratio of the sizes
    # and use that percentage to undersample the safe loans.
    percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

    risky_loans = risky_loans_raw
    safe_loans = safe_loans_raw.sample(frac=percentage,  random_state=1)

    # Append the risky_loans with the downsampled version of safe_loans
    loans_data = risky_loans.append(safe_loans)
    
    # one hot encode
    cols_cat = loans_data.select_dtypes(include = ["object"]).columns
    cols_num = loans_data.select_dtypes(include = ["int64", "float64"]).columns
    loans_data_cat = one_encoder(loans_data, cols_cat)
    loans_data_num = loans_data[cols_num]
    
    loans_data = loans_data_cat.join(loans_data_num)
    loans_data = loans_data.fillna(0)
    train_data, val_data = train_test_split(loans_data, train_size = 0.8, random_state = 1)
    
    #train model
    decision_tree_model = tree.DecisionTreeClassifier(max_depth=6)
    small_model = tree.DecisionTreeClassifier(max_depth=2)
    small_model.fit(train_data[features], train_data[target])
    tree.export_graphviz(small_model, out_file='tree.dot')
    decision_tree_model.fit(train_data[features], train_data[target])

    #Quiz Question: What percentage of the predictions on 
    #sample_validation_data did decision_tree_model get correct?
    val_safe_loans = val_data[val_data[target] == 1]
    val_risky_loans = val_data[val_data[target] == -1]

    sample_val_data_risky = val_risky_loans[0:2]
    sample_val_data_safe = val_safe_loans[0:2]

    sample_val_data = sample_val_data_safe.append(sample_val_data_risky)
    prediction = decision_tree_model.predict(sample_val_data[features])
    print "-------Questions--------\n"
    print prediction, "\n"
    print "percentage of the correct predictions on sample_validation_data:",\
         1.*sum([1 for i in range(len(prediction)) if sample_val_data[target].values[i] \
                 == prediction[i]])/len(prediction), "\n"

    #Quiz Question: Which loan has the highest probability of being classified 
    #as a safe loan?
    probs = decision_tree_model.predict_proba(sample_val_data[features])
    print "Probalibity of prediction: ", probs, "\n"

    #Quiz Question: Notice that the probability preditions are the exact same 
    #for the 2nd and 3rd loans. Why would this happen?



    #Quiz Question: Based on the visualized tree, what prediction would you 
    #make for this data point (according to small_model)? (If you don't have 
    #Graphviz, you can answer this quiz question by executing the next part.)


    #Quiz Question: What is the accuracy of decision_tree_model on the 
    #validation set, rounded to the nearest .01?
    score_small_model = small_model.score(val_data[features], val_data[target])
    score_tree_model = decision_tree_model.score(val_data[features], val_data[target])
    print "Accuracy score for small model- %.2f and tree model- %.2f"%(score_small_model, score_tree_model), "\n"

    #Quiz Question: How does the performance of big_model on the validation set 
    #compare to decision_tree_model on the validation set? Is this a sign of overfitting?
    big_model = tree.DecisionTreeClassifier(max_depth=10)
    big_model.fit(train_data[features], train_data[target])
    score_big_model = big_model.score(val_data[features], val_data[target])
    print "Accuracy for big model %.2f"%score_big_model


   #Quiz Question: Let's assume that each mistake costs us money: 
   #a false negative costs $10,000, while a false positive positive costs $20,000. 
   #What is the total cost of mistakes made by decision_tree_model on validation_data?
    preds = decision_tree_model.predict(val_data[features])
    y = val_data[target].values
    cost = 0
    for i in range(len(y)):
        if preds[i] != y[i]:
            if preds[i] == -1:
                cost += 10000
            else: cost += 20000
    print "Total cost for prediction: ", cost