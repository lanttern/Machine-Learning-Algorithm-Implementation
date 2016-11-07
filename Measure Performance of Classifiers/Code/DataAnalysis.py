# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 20:43:25 2016

@author: zhihuixie
"""
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import matplotlib

class DataAnalysis():
    """
    This class is used for data exploration of classification dataset.
    """
    def __init__(self, data, is_voicedata = True):
        """
        import data as pandas dataframe
        """
        self.data = data
        self.is_voicedata = is_voicedata
        
        
    def data_exploration(self):
        """
        This functon explores basic characteristics of dataset incuding number 
        of data points, number of features, number of missing values and number
        of datapoints in each class.
        """

        # compute number of labels in each class
        if not self.is_voicedata:
            number_of_class_0 = self.data[self.data["Self-defined label"] == 0].shape[0]
            number_of_class_1 = self.data[self.data["Self-defined label"] == 1].shape[0]
            new_data = self.data.drop("Self-defined label",  axis = 1)
        else:
            number_of_class_0 = self.data[self.data["label"] == "male"].shape[0]
            number_of_class_1 = self.data[self.data["label"] == "female"].shape[0]
            new_data = self.data.drop("label", axis = 1)
        
        # compute number of data points and features
        (number_of_data_points, number_of_features) = (new_data.shape[0], \
                                                           new_data.shape[1])        
        # compute number of missing values
        number_of_missing_value = self.data.isnull().values.ravel().sum()

        return (number_of_data_points, number_of_features,  number_of_class_0,\
                number_of_class_1, number_of_missing_value)
    
    
    def feature_analysis(self):
        """
        Make a plot to visulize important features to separate labels
        """

        if self.is_voicedata:
            # explore all paired scatter plots
            seaborn.set_context("poster")
            plt.figure(figsize = (10,8))
            plot_all = seaborn.pairplot(self.data, hue = "label")
            plt.suptitle("Feature analysis for voice dataset - all features")
            plot_all.savefig("../Figures/voice_exploration.png", bbox_inches="tight")
            # explore paired scatter plots with selected features
            plt.figure(figsize = (10,8))
            plot_selected = seaborn.pairplot(self.data[["skew","kurt", "meanfun", \
                         "meanfreq", "IQR", "label"]], hue = "label")
            plt.suptitle("Feature analysis for voice dataset - selected features")
            plot_selected.savefig("../Figures/voice_exploration_selected.png", bbox_inches="tight")
        else:
            seaborn.set_context("poster")
            plt.figure(figsize = (10,8))
            # explore all paired scatter plots
            plot_all = seaborn.pairplot(self.data, hue = "Self-defined label")
            plt.suptitle("Feature analysis for EEG dataset - all features")
            plot_all.savefig("../Figures/EEG_exploration.png", bbox_inches="tight")
            # explore paired scatter plots with selected features
            plt.figure(figsize = (10,8))
            plot_selected = seaborn.pairplot(self.data[["Delta","Theta","Alpha 1",\
                           "Beta 2", "Gamma1", "Self-defined label"]], hue = "Self-defined label")
            plt.suptitle("Feature analysis for EEG dataset - selected features")
            plot_selected.savefig("../Figures/EEG_exploration_selected.png", bbox_inches="tight")
    
if __name__ == "__main__":
    # read dataset 1 - Voice Data for Gender Recognition
    data_1 = pd.read_csv("../Dataset/voice.csv")
    # read dataset 2 - EEG Data for Brain Confusion Prediction
    data_2 = pd.read_csv("../Dataset/EEG data.csv")
    # drop unused columnes: predefined label, subject ID and video ID
    data_2.drop("predefined label", inplace = True, axis = 1)
    data_2.drop("subject ID",  inplace = True, axis = 1)
    data_2.drop("Video ID",  inplace = True, axis = 1)
    
    # read data into DataAnalysis class
    da_1 = DataAnalysis(data_1)
    da_2 = DataAnalysis(data_2, is_voicedata = False)
    
    # check basic information of dataset
    print "The first 3 lines of voice dataset: "
    print data_1.head(3), "\n"
    print "The basic information of voice dataset: "
    basic_info = da_1.data_exploration()
    print "The number of data points: ", basic_info[0]
    print "The number of features: ", basic_info[1]
    print "The number of class labeled as 0: ", basic_info[2]
    print "The number of class labeled as 1: ", basic_info[3]
    print "The number of missing value: ", basic_info[4], "\n"
    if basic_info[4] > 0:
        data_1.fillna(0)
    
    print "The first 3 lines of EEG dataset: "
    print data_2.head(3), "\n"    
    print "The basic information of EEG dataset: "
    basic_info = da_2.data_exploration()
    print "The number of data points: ", basic_info[0]
    print "The number of features: ", basic_info[1]
    print "The number of class labeled as 0: ", basic_info[2]
    print "The number of class labeled as 1: ", basic_info[3]
    print "The number of missing value: ", basic_info[4], "\n"
    # replace missing value with blank space
    if basic_info[4] > 0:
        data_2.fillna(0)
    # feature analysis for voice dataset     
    da_1.feature_analysis()
    # feature analysis for stock market dataset
    da_2.feature_analysis()
    
    # versions of packages
    print "The version of pandas: ", pd.__version__
    print "The version of seaborn: ", seaborn.__version__
    print "The version of matplotlib: ", matplotlib.__version__
    
    
        
        