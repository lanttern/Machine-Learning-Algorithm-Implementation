# -*- coding: utf-8 -*-
"""
Created on April 6 20:01:35 2016

@author: zhihuixie
"""
import pandas as pd
from zipfile import ZipFile
import json
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

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
    features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

    target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

    # Extract the feature columns and target column and remove missing data point
    loans = loans[features + [target]].dropna()
    # one hot encode
    cols_cat = loans.select_dtypes(include = ["object"]).columns
    cols_num = loans.select_dtypes(include = ["int64", "float64"]).columns
    #loans_data_cat = one_encoder(loans, cols_cat)
    loans_data_cat = pd.get_dummies(loans[cols_cat])
    loans_data_num = loans[cols_num]
    
    loans_data = loans_data_cat.join(loans_data_num)
    new_features = loans_data.columns
    
    #split train validation dataset
    with open("module-8-assignment-1-train-idx.json", "r") as f1, \
         open("module-8-assignment-1-validation-idx.json", "r") as f2:
             train_idx = json.load(f1)
             val_idx = json.load(f2)
    train_data = loans_data.iloc[train_idx]
    val_data = loans_data.iloc[val_idx]
    
    val_safe_loans = val_data[val_data[target] == 1]
    val_risky_loans = val_data[val_data[target] == -1]

    sample_val_data_risky = val_risky_loans[0:2]
    sample_val_data_safe = val_safe_loans[0:2]

    sample_val_data = sample_val_data_safe.append(sample_val_data_risky)
    y = sample_val_data[target].values
    print "The real output: ", y, "\n"
    
    model_5 = GradientBoostingClassifier(max_depth=6, n_estimators=5)
    model_5.fit(train_data[new_features[:-1]], train_data[target])
    preds = model_5.predict(sample_val_data[new_features[:-1]])
    percent = 1.0*sum([1 for i in range(len(y)) if y[i] == preds[i]])/len(y)
    
    #Quiz question: What percentage of the predictions on sample_validation_data
    #did model_5 get correct?
    print "Prediction for sample data: ", preds, "\n"
    print "Percentage of corrected prediction: %f"%percent, "\n"
    
    #Quiz Question: Which loan has the highest probability of being classified as a safe loan?
    print "Probability of prediction: ", model_5.predict_proba(sample_val_data[new_features[:-1]]), "\n"
    
    score = model_5.score(val_data[new_features[:-1]], val_data[target])
    print "Score for validation data: ",score, "\n"
    
    #Quiz question: What is the number of false positives and false negative \
    #on the validation_data?
    val_y = val_data[target].values
    val_preds = model_5.predict(val_data[new_features[:-1]])
    fal_pos = sum([1 for i in range(len(val_y)) if val_preds[i] == 1 and \
                  val_y[i] != val_preds[i]])
    fal_neg = sum([1 for i in range(len(val_y)) if val_preds[i] == -1 and \
                  val_y[i] != val_preds[i]])
    print "number of false positive and false negative are %d, %d"%(fal_pos, fal_neg), "\n"
    
    #Quiz Question: Using the same costs of the false positives and false negatives,
    #what is the cost of the mistakes made by the boosted tree model (model_5) as 
    #evaluated on the validation_set?
    cost = fal_pos*20000 + fal_neg*10000
    print "The total cost is: ", cost, "\n"
    
    #Quiz question: What grades are the top 5 loans and 5 loans 
    #(in the validation_data) with the lowest probability ?
    probs = model_5.predict_proba(val_data[new_features[:-1]])
    probs_safe = probs[:, 1]
    val_data.index = range(val_data.shape[0])
    probs_zip = zip(probs_safe, val_data.index.values)
    probs_sorted = sorted(probs_zip, key = lambda x: x[0])
    print "The top 5 loans: ", probs_sorted[-5:], "\n"
    print "The lowest 5 loans: ", probs_sorted[:5], "\n"
    sorted_index = [j for (i,j) in probs_sorted[-5:]]
    print loans_data.iloc[sorted_index], "\n"
    
    #Effects of adding more trees
    """
    Train models with 10, 50, 100, 200, and 500 trees. Use the n_estimators 
    parameter to control the number of trees. Remember to keep max_depth = 6.

    Call these models model_10, model_50, model_100, model_200, and model_500, 
    respectively. This may take a few minutes to run.

    Compare accuracy on entire validation set
    Evaluate the accuracy of the 10, 50, 100, 200, and 500 tree models on the 
    validation_data.
    Quiz Question: Which model has the best accuracy on the validation_data?
    Quiz Question: Is it always true that the model with the most trees will 
    perform best on test data?
    """
    trees = [10, 50, 100, 200, 500]
    training_errors, validation_errors = [], []
    for tree in trees:
        model_tree = GradientBoostingClassifier(max_depth=6, n_estimators=5)
        model_tree.fit(train_data[new_features[:-1]], train_data[target])
        train_score = model_tree.score(train_data[new_features[:-1]], train_data[target])
        score = model_tree.score(val_data[new_features[:-1]], val_data[target])
        training_errors.append(1-train_score)
        validation_errors.append(1-score)
        print "The accuracy for tree %d is %f"%(tree, score), "\n"
        
    #Quiz question: Does the training error reduce as the number of trees increases?
    #Quiz question: Is it always true that the validation error will reduce as 
    #the number of trees increases?
        
    plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
    plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

    make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')
    
    
    
    
    