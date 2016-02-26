# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:20:35 2016

@author: zhihuixie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
    
def get_matrix(df, features, output):
    """
    transform data from pandas dataframe to matrix
    """
    #add a constant column as coefficient for w0
    df["constant"] = 1.0
    feature_x, output_y = df[features].astype(float), df[output].astype(float)
    return np.array(feature_x),np.array(output_y)
    
def visualization(x, y1, y2):
     """
     visualize the prediction results
     color real output as blue, prediction as red
     """
     plt.figure()
     plt.plot(x, y1, ".", color = "blue")
     plt.plot(x, y2, color = "red")
     plt.legend(["kernel", "simple average"])
     plt.title("Compare kernel function and simple average")

class KnnRegression():
    """
    This class implement polynomial linear regression algorithm  
    """
    def __init__(self, train_feature_x, train_output_y):
        """
        four parameters: as pandas dataframe series
        """
        self.train_feature_x = train_feature_x
        self.train_output_y = train_output_y
        self.initial_weights = np.zeros(self.train_feature_x.shape[1])
    
    def normalize_features(self, normalize = True):
        """
        normalizes columns of a given feature matrix
        """
        if normalize:
            norms = np.linalg.norm(self.train_feature_x, axis = 0)
            normalized_features = self.train_feature_x/norms
        return normalized_features, norms
        
    def euc_dist(self, features_query):
        """
        compute Euclidean distance between the query house 
        and the house of the training set?
        """            
        train_features, _ = self.normalize_features()
        diff = train_features - features_query
        dist = np.sqrt(np.sum(diff**2, axis = 1))
        return dist
        
    def k_nearest_neighbors(self, k, features_query):
        """
        returns the indices of the k closest training houses. 
        """
        dist = self.euc_dist(features_query)
        neighbors = np.argsort(dist)[:k]
        return neighbors
        
    def prediction(self, x, k, kernel = False):
        """
        predict the value of each and every house in a query set
        use gaussian kernel for prediction if kernel is true
        """
        num_x = x.shape[0]
        neighbors_matrix = np.array([self.k_nearest_neighbors(k,x[i]) \
                                                      for i in range(num_x)])
        if not kernel:
            predictions = np.array([sum(self.train_output_y[neighbors_matrix[j]])/k\
                            for j in range(num_x)])
        else:
            weights = np.array([np.exp(-self.euc_dist(x[n])\
                     [np.argsort(self.euc_dist(x[n]))[:k]]/k) \
                     for n in range(num_x)])
            predictions = np.array([np.dot(self.train_output_y[neighbors_matrix[m]],\
                weights[m])/sum(weights[m]) for m in range(num_x)])
        return predictions
                 
if __name__ == "__main__":
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, \
    'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, \
    'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, \
    'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, \
    'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, \
    'sqft_lot':int, 'view':int}
    
    #load data
    #train data
    zf1 = zipfile.ZipFile("kc_house_data_small_train.csv.zip")
    df_train = pd.read_csv(zf1.open("kc_house_data_small_train.csv"), dtype=dtype_dict) 
    #test data
    zf2 = zipfile.ZipFile("kc_house_data_small_test.csv.zip")
    df_test = pd.read_csv(zf2.open("kc_house_data_small_test.csv"), dtype=dtype_dict)
    #valid data
    zf3 = zipfile.ZipFile("kc_house_data_small_validation.csv.zip")
    df_valid = pd.read_csv(zf3.open("kc_house_data_validation.csv"), dtype=dtype_dict) 
    #featureas
    features = df_valid.columns.values[3:]
    output = "price"
    train_features, train_output = get_matrix(df_train, features, output)
    test_features, test_output = get_matrix(df_test, features, output)
    valid_features, valid_output = get_matrix(df_valid, features, output)

    #normalization
    model =KnnRegression(train_features, train_output)
    train_features,norms = model.normalize_features()
    test_features = test_features/norms
    valid_features = valid_features/norms
    
    #Q1:What is the Euclidean distance between the query house and the 10th 
    #house of the training set?
    euc_0 = model.euc_dist(test_features[0])
    print "Q1: Euclidean distance between the query house and the 10th house:",\
           euc_0[9], "\n"
           
    #Q2:Among the first 10 training houses, which house is the closest to the query house?
    print "Q2: Among the first 10 training houses, the closest to the query house", \
           np.argsort(euc_0[:10])[0], "\n"

    #Q3:Take the query house to be third house of the test set (features_test[2]). 
    #What is the index of the house in the training set that is closest to this 
    #query house?
    neighbor = model.k_nearest_neighbors(1, test_features[2])
    print "Q3: Closest neighbor to 3rd house of the test set:", neighbor, "\n"
           
    #Q4:What is the predicted value of the query house based on 1-nearest 
    #neighbor regression?
    pred_k1 = model.prediction(x = np.array([test_features[2]]), k=1)
    print "Q4:The predicted value of the query house based on 1-NN:", pred_k1, "\n"
    
    #Q5:Take the query house to be third house of the test set (features_test[2]).
    #What are the indices of the 4 training houses closest to the query house?
    neighbors = model.k_nearest_neighbors(4, test_features[2])
    print "Q5: Closest neighbors to 3rd house of the test set:", neighbors, "\n"
    
    #Q6:Again taking the query house to be third house of the test set 
    #(features_test[2]), predict the value of the query house using k-nearest 
    #neighbors with k=4 and the simple averaging method described and implemented above.
    pred_k4 = model.prediction(x = np.array([test_features[2]]), k=4)
    print "Q6:The predicted value of the query house based on 4-NN:", pred_k4, "\n"
    
    #Q7:Make predictions for the first 10 houses in the test set, using k=10.
    #What is the index of the house in this query set that has the lowest 
    #predicted value? What is the predicted value of this house?
    preds = model.prediction(x = test_features[:10], k=10)
    index = np.argsort(preds)
    print "Q7: the lowest predicted values:", preds[index[0]], "\n"
    
    #Q8:What is the RSS on the TEST data using the value of k found above? 
    #To be clear, sum over all houses in the TEST set.
    best_rss = float("inf")
    best_k = 0
    for k in range(1,16):
        preds = model.prediction(x = valid_features, k = k)
        rss = sum((preds - valid_output)**2)
        if rss < best_rss:
            best_rss = rss
            best_k = k
    print "The best rss and best k for validation set:", (best_rss, best_k), "\n"
    test_preds = model.prediction(x = test_features, k = best_k, kernel = True)
    test_rss = sum((test_preds - test_output)**2)
    print "Q8: The RSS on the TEST data with best k:", test_rss, "\n"
    
    #compare gaussian kernel and constant average
    y1, y2 = [], []
    for k in range(1,51):
        valid_preds1 = model.prediction(x = valid_features, k = k, kernel = True)
        valid_rss1 = sum((valid_preds1 - valid_output)**2)
        y1.append(valid_rss1)
        valid_preds2 = model.prediction(x = valid_features, k = k, kernel = False)
        valid_rss2 = sum((valid_preds2 - valid_output)**2)
        y2.append(valid_rss2)
    visualization(range(1,51), y1, y2)