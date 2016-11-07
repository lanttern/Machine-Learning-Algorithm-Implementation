# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:51:40 2016

@author: zhihuixie
"""
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn import random_projection as RP
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kurtosis
import pandas as pd
import numpy as np


import time
import sklearn
import pandas
import numpy
import matplotlib
import scipy


class clustering():
    """
    this calss is applied to analyze the performance of two clustering algorithms:
    1) K-means, 2) Expectation–maximization
    """
    def __init__(self, X, y):
        """
        setup variables: X - features, y - labels
        """
        self.X = X
        self.y = y
        
    def k_means_analysis(self, score_func, title, xlabel, ylabel, n_runs = 5, is_inertia = True,
                        n_clusters_range = np.linspace(2, 100, 10).astype(np.int)):
        """
        this method analyzes the performance of KMeans clustering with different metrics
        """
        scores = np.zeros((len(n_clusters_range), n_runs))
        for i in range(len(n_clusters_range)):
            clf = KMeans(n_clusters = n_clusters_range[i])
            clf.fit(self.X)
            for j in range(n_runs):
                if is_inertia:
                    scores[i,j] = score_func(self.X, clf.labels_)
                else:
                    y_pred = clf.predict(self.X)
                    scores[i, j] = score_func(self.y, y_pred)
        self.make_plots(scores,score_func, n_clusters_range, 
                        title, xlabel, ylabel)

    def em_analysis(self, score_func, title, xlabel, ylabel, n_runs = 5,is_inertia = True, 
                    n_components_range = np.linspace(2, 100, 10).astype(np.int)):
        """
        this method analyzes the performance of EM clustering with different metrics
        """
        scores = np.zeros((len(n_components_range), n_runs))
        for i in range(len(n_components_range)):
            clf = GaussianMixture(n_components = n_components_range[i])
            clf.fit(self.X)
            for j in range(n_runs):
                y_pred = clf.predict(self.X)
                if is_inertia:
                    scores[i,j] = clf.bic(self.X)
                else:
                    scores[i, j] = score_func(self.y, y_pred)
        self.make_plots(scores,score_func, n_components_range, 
                        title, xlabel, ylabel)
    
    def make_plots(self, scores, score_func, x_axis, title, xlabel, ylabel):
        
        """
        this method makes plot to visualize the performance of clustering algorithms
        """
        
        print("Computing %s for %d params of n_clusters"
                % (score_func.__name__, len(x_axis)))
        plt.errorbar(x_axis, np.mean(scores, axis=1), yerr = scores.std(axis=1), fmt = "o-")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    def final_plots(self, score_funcs, title, xlabel,ylabel, 
                    n_range = np.linspace(2, 100, 10).astype(np.int), is_inertia = True, alg = "k_means"):        
        """
        this method is to generate the final plot for the analysis
        """
        plt.figure(figsize = (8,8))
        names = []
        # n_range = np.array([2, 10, 12, 23, 34, 45, 56, 67, 78, 89, 100])
        for score_func in score_funcs:
            t0 = time.time()
            if alg == "k_means":
                self.k_means_analysis(score_func, title, xlabel, ylabel, 
                            is_inertia = is_inertia, n_clusters_range = n_range)
            elif alg == "em":
                self.em_analysis(score_func, title, xlabel, ylabel, 
                        is_inertia = is_inertia, n_components_range = n_range)
            else:
                print "The algorithm is not inclued in this analysis."
                break
            names.append(score_func.__name__)
            print("done in %0.3fs" % (time.time() - t0))
        plt.legend(names, bbox_to_anchor=(0., 1.05, 1., .105),loc=3, mode="expand",
                   borderaxespad=0.)
        plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
        
    def parameter_optimization(self, estimator, param_name, params, score_func, 
                               alg = "k-mean", is_inertia = True):
        """
        this method optimize parameters and return the score with optimized parameter
        """
        score = 0
        best_score = 0
        best_param = params[0]
        print "Start searching parameters ...."
        for i in range(len(params)):
            estimator.set_params(**{param_name: params[i]})
            estimator.fit(self.X)
            if is_inertia:
                if alg == "k-means":
                    score = score_func(self.X, estimator.labels_)
                elif alg == "em":
                    score = estimator.bic(self.X)
            else:
                y_pred = estimator.predict(self.X)
                score = score_func(self.y, y_pred)
            print param_name + ": ", params[i], "with score: ", score
            if score > best_score:
                best_score = score
                best_param = params[i]
        return best_param,best_score
    
    def optimization_results(self, clf, score_funcs, param_grid, alg = "k-means",
                             is_inertia=True):
        """
        this method is to print out results for parameter optimization
        """
        for key in param_grid:
            for score_func in score_funcs:
                if alg == "k-means":
                    print "Results for " + score_func.__name__ + " with parameter: "+ key
                else:
                    print "Results for BIC score" + " with parameter: "+ key
                print "--------------------------"
                best_param, best_score = self.parameter_optimization(clf, key, 
                          param_grid[key], score_func, alg = alg,is_inertia = is_inertia)
                print "Best parameter ", best_param, "and score: ", best_score
                print "--------------------------\n"
    
class dimen_reduction():
    """
    this class is applied to analyze the dimensionality reduction algorithms:
    1) Principal Component Analysis (PCA), 2) Independent Component Analysis (ICA),
    3) Randomized Projections, 4) Linear Discriminant Analysis (LDA)
    """
    def __init__(self, X, y):
        """
        setup variables: X - features, y - labels
        """
        self.X = X
        self.y = y
        
    def plot_2d(self, estimator, title, xlabel, ylabel):
        """
        this method generates 2D plot for dimensionality reduction
        """
        # Fit the model on the numeric columns from earlier.
        plot_columns = estimator.fit_transform(self.X)
        # Make a scatter plot, shaded according to cluster assignment.
        plt.figure(figsize = (8,8))
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=self.y, cmap = cm.brg_r)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
        
    def plot_3d(self, estimator, title, xlabel, ylabel, zlabel):
         """
         this method generates 3D plot for dimensionality reduction
         """
         fig = plt.figure(figsize=(8, 8))
         ax = Axes3D(fig, elev=-150, azim=110)
         X_reduced = estimator.fit_transform(self.X)
         ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=self.y,
                    cmap=cm.brg_r)
         ax.set_title(title)
         ax.set_xlabel(xlabel)
         ax.w_xaxis.set_ticklabels([])
         ax.set_ylabel(ylabel)
         ax.w_yaxis.set_ticklabels([])
         ax.set_zlabel(zlabel)
         ax.w_zaxis.set_ticklabels([])
         plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
         
    def eig_plot(self, estimator, x_axis, title, xlabel, ylabel, dataset_name):
        """
        this method generates eig value distribution graph
        """
        plt.figure(figsize=(8, 8))
        estimator.fit_transform(self.X)
        eigenvalues = estimator.explained_variance_
        if dataset_name == "voice dataset":
            plt.plot(x_axis, np.log10(eigenvalues), "o-")
            plt.axhline(y = 0, color = "r")  
        else:
            plt.plot(x_axis, eigenvalues, "o-")
            plt.axhline(y = 1, color = "r")  
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
        
    def kurt_plot(self, x_axis, title, xlabel, ylabel):
        """
        this method creates kurt score distribution graph
        """
        plt.figure(figsize=(8, 8))
        kurt_score = []
        for i in x_axis:
            ica = FastICA(n_components = i)
            fitted_data_ica = ica.fit_transform(self.X)
            kurt = kurtosis(fitted_data_ica)
            kurt_score.append(np.sum(kurt))
        # print kurt_score
        plt.plot(x_axis, kurt_score, "o-")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
        
    def rec_plot(self, x_axis, title, xlabel, ylabel, est = "PCA", 
                 is_fixed = False):
        """
        this method creates reconstruction plots for PCA, ICA and RP
        """
        plt.figure(figsize=(8, 8))
        loss_score = []
        for i in x_axis:
            if est == "PCA":
                dr_est = PCA(i)
            elif est == "ICA":
                dr_est = FastICA(i)
            elif est == "TruncatedSVD":
                dr_est = TruncatedSVD(i)
            elif est == "RP":
                if is_fixed:
                    dr_est = RP.GaussianRandomProjection(2)
                else:
                    dr_est = RP.GaussianRandomProjection(i)
            else:
                print "Error: incrrect estimator"
            X_dr= dr_est.fit_transform(self.X)
            # mean squared error to measure the reconstruction error
            X_projected = dr_est.inverse_transform(X_dr)
            loss = ((self.X - X_projected)**2).mean()
            loss_score.append(loss)
        # print loss_score
        plt.plot(x_axis, loss_score, "o-")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
        
    def clustering_plot(self, score_func, est = "k_means", dr = "PCA", 
                        n_features = 20, dataset_name = "voice dataset"):
        """
        this method generate plots for KMeans or EM with dimensionality reduction
        """
        scores = np.zeros((n_features, 1))
        if est == "k_means":
            if dataset_name == "voice dataset":
                clf = KMeans(n_clusters = 2)
            else:
                clf = KMeans(n_clusters = 10)
        elif est == "em":
            if dataset_name == "voice dataset":
                clf = GaussianMixture(n_components = 2)
            else:
                clf = GaussianMixture(n_components = 10)
        else:
            print "Incorrect estimator"
            return
        clf.fit(self.X)
        est_y = clf.predict(self.X)
        for i in range(1, n_features + 1):
            if dr == "PCA":
                dr_est = PCA(i)
            elif dr == "ICA":
                dr_est = FastICA(i)
            elif dr == "TruncatedSVD":
                dr_est = TruncatedSVD(i)
            elif dr == "RP":
                dr_est = RP.GaussianRandomProjection(i)
            else:
                print "Error: incrrect estimator"
                return
            dr_X = dr_est.fit_transform(self.X)
            clf.fit(dr_X)
            est_dr_y = clf.predict(dr_X)
            scores[i-1,0] = score_func(est_y, est_dr_y)
        return scores
        
class neural_network_analysis():
      """
      this class is applied to analyze the neural network performance with 
      features generated by dimensionality reduction or clustering
      """
      def __init__(self, X, y):
        """
        setup variables: X - features, y - labels
        """
        self.X = X
        self.y = y
      
      def performance(self, clf, dr_ests, X_all, y_all, X_train_1, y_train_1, X_val_1, 
                        y_val_1, title,xlabel, ylabel,colors, n_features, is_dr = True):
        
          """
          this method generates validation plots with ann classifier and different
          dimensionality reduction
          """
          plt.figure(figsize = (8,8))
          t_dr = 0
          t_ann = 0
          accuracy = 0
          precision = 0
          recall = 0
          f1 = 0
          for n in range(5):
              t0 = time.time()
              clf.fit(X_train_1, y_train_1)
              y_pred = clf.predict(X_val_1)
              t1 = time.time()
              accu = metrics.accuracy_score(y_val_1, y_pred) 
              pre = metrics.precision_score(y_val_1, y_pred)
              rec = metrics.recall_score(y_val_1, y_pred)
              f = metrics.f1_score(y_val_1, y_pred)
              t_ann += t1-t0
              accuracy += accu
              precision += pre
              recall += rec
              f1 += f
          plt.axhline(y=accuracy/5, color = "r", label = "Original baseline")
          print "Optimized ann baseline: "
          print "----------------------"
          print "accuracy: %f, precision: %f, recall: %f, f1: %f, ann time: \
                %f sec"%(accuracy/5, precision/5, recall/5, f1/5, t_ann/5)
          print "----------------------\n"
          for i in range(len(dr_ests)):
              if dr_ests[i][0] == "TruncatedSVD":
		      n_features = n_features - 1
              scores = []
              for j in range(1,n_features + 1):
                  t_dr = 0
                  t_ann = 0
                  accuracy = 0
                  precision = 0
                  recall = 0
                  f1 = 0
                  for k in range(5):
                      t0 = time.time()
                      dr = dr_ests[i][1](j)
                      dr.fit(X_train_1)
                      if is_dr:
                          X_train = dr.transform(X_train_1)
                          X_val = dr.transform(X_val_1)
                      else:
                          df_train = pd.DataFrame()
                          for l in range(1, j+1):
                              X_train_temp = dr.predict(X_all)
                              df_train["%s"%l] = X_train_temp
                          X_train = df_train.values
                          X_train, X_val, y_train_1, y_val_1 = train_test_split\
                                    (X_train,y_all, test_size=0.3, random_state=42)
                      t1 = time.time()
                      clf.fit(X_train, y_train_1)
                      y_pred = clf.predict(X_val)
                      t2 = time.time()
                      accu = metrics.accuracy_score(y_val_1, y_pred)
                      pre = metrics.precision_score(y_val_1, y_pred)
                      rec = metrics.recall_score(y_val_1, y_pred)
                      f = metrics.f1_score(y_val_1, y_pred)
                      t_dr += t1-t0
                      t_ann += t2-t1
                      accuracy += accu
                      precision += pre
                      recall += rec
                      f1 += f
                  print "%s with %d components:"%(dr_ests[i][0], j)
                  print "----------------------"
                  print "accuracy: %f, precision: %f, recall: %f, f1: %f, \
                         transformation time: %f sec, ann time: %f sec"\
                         %(accuracy/5, precision/5, recall/5, f1/5, t_dr/5, t_ann/5)
                  print "----------------------\n"
                  scores.append(accuracy/5)
              plt.plot(range(1,n_features + 1), scores, "o-", label = dr_ests[i][0], color = colors[i])
          plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
          plt.title(title)
          plt.xlabel(xlabel)
          plt.ylabel(ylabel)
          plt.ylim(ymax = 1.0)
          plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
      
      def dim_reduction_performance(self):
          pass
    
    
if __name__ == "__main__":
    
    def analysis(X, y, score_funcs, score_funcs_inertia, n_features, dataset_name):
        """
        This method analyze the performance of two cluster algorithms and four 
        dimentionality reduction algorithms
        """
        
        # measure performance of K-means and EM  
        # explanations of your methods: How did you choose k?
        # a description of the kind of clusters that you got.
        # analyses of your results. Why did you get the clusters you did? Do they 
        # make "sense"? If you used data that already had labels (for example data 
        # from a classification problem from assignment #1) did the clusters line 
        # up with the labels? Do they otherwise line up naturally? Why or why not? 
        # Compare and contrast the different algorithms.
        n_range = np.linspace(2, 100, 10).astype(np.int)          
        cluster = clustering(X,y)
        
        # plot with internal metrics
        title = "KMeans internal measure scores with different clusters (%s)"%dataset_name
        xlabel = "Number of clusters"
        ylabel = "Score value"
        cluster.final_plots(score_funcs_inertia, title, xlabel, ylabel, n_range = n_range)
        # plot with external metrics
        title = "KMeans external measure scores with different clusters (%s)"%dataset_name
        cluster.final_plots(score_funcs, title, xlabel, ylabel, n_range = n_range, 
                            is_inertia = False)
        
        # plot with internal metrics
        title = "EM internal measure scores with different components (%s)"%dataset_name
        xlabel = "Number of components"
        ylabel = "BIC score" 
        cluster.final_plots(score_funcs_inertia, title, xlabel, ylabel, 
                                            n_range = n_range, alg = "em")
        # plot with external metrics
        ylabel = "Score value" 
        title = "EM external measure scores with different components (%s)"%dataset_name
        cluster.final_plots(score_funcs, title, xlabel,ylabel, n_range = n_range, 
                                              alg = "em", is_inertia = False)

        # optimize parameters for K-means and EM
        # What sort of changes might you make to each of those algorithms to improve 
        # performance? How much performance was due to the problems you chose? Be 
        # creative and think of as many questions you can, and as many answers as 
        # you can. Take care to justify your analysis with data explictly.
        if dataset_name == "voice dataset":
            clf = KMeans(n_clusters = 2)
        else:
            clf = KMeans(n_clusters = 10)
        clf.fit(X)
        s1 = metrics.silhouette_score(X, clf.labels_)
        print "The internal score for KMeans default parameters: ", s1
    
        param_grid = {"max_iter":[100,200,400,800], "n_init":[5,10,15,20], 
                  "init" : ["random", "k-means++"], 
                  "algorithm" : ["auto", "full", "elkan"],
                  "tol": np.logspace(-10,-1,4)}

        cluster.optimization_results(clf, score_funcs_inertia, param_grid)
              
        y_pred1 = clf.fit_predict(X)
        for score_func in score_funcs:
            score = score_func(y, y_pred1)
            print "The external score for KMeans with %s for default parameters: "%(score_func.__name__), score
        cluster.optimization_results(clf, score_funcs, param_grid, is_inertia = False)
        if dataset_name == "voice dataset":
            clf = GaussianMixture(n_components = 2)
        else:
            clf = GaussianMixture(n_components = 10)
        clf.fit(X)
        s1 = clf.bic(X)
        print "The internal score for EM with default parameters: ", s1
    
        param_grid = {"max_iter":[100,200,400,800], "n_init":[5,10,15,20], 
                  "init_params" : ["random", "kmeans"], 
                  "covariance_type" : ["full", "tied", "diag","spherical"],
                  "tol": np.logspace(-10,-1,4)}

        cluster.optimization_results(clf, score_funcs_inertia, param_grid, alg = "em")
              
        y_pred1 = clf.predict(X)
        for score_func in score_funcs:
            score = score_func(y, y_pred1)
            print "The external score for EM with %s for default parameters: "%(score_func.__name__), score
        cluster.optimization_results(clf, score_funcs, param_grid, alg = "em",
                                       is_inertia = False)  
        
        # analyze dimensionality reduction
        # Can you describe how the data look in the new spaces you created 
        # with the various aglorithms?    

        dimensionality_reduction = dimen_reduction(X, y)
        # Create a PCA model.
        pca_2 = PCA(2)
        title = "Two PCA directions (%s)"%dataset_name
        xlabel = "1st eigenvector"
        ylabel = "2nd eigenvector"
    
        dimensionality_reduction.plot_2d(pca_2, title, xlabel, ylabel)
    
        # To getter a better understanding of interaction of the dimensions
        # plot the first three PCA dimensions
        pca_3 = PCA(3)
        title = "Three PCA directions (%s)"%dataset_name
        zlabel = "3rd eigenvector"
        dimensionality_reduction.plot_3d(pca_3, title, xlabel, ylabel, zlabel)
    
    
        # Create a ICA model
        ica_2 = FastICA(2)
        title = "Two ICA directions (%s)"%dataset_name
        dimensionality_reduction.plot_2d(ica_2, title, xlabel, ylabel)
    
        ica_3 = FastICA(3)
        title = "Three ICA directions (%s)"%dataset_name
        dimensionality_reduction.plot_3d(ica_3, title, xlabel, ylabel, zlabel)
    
        # Create BP model
        rp_2 = RP.GaussianRandomProjection(2)
        title = "Two Gaussian random projection directions (%s)"%dataset_name
        dimensionality_reduction.plot_2d(rp_2, title, xlabel, ylabel)
    
        rp_3 = RP.GaussianRandomProjection(3)
        title = "Three Gaussian random projection directions (%s)"%dataset_name
        dimensionality_reduction.plot_3d(rp_3, title, xlabel, ylabel, zlabel)
    
        # Create TruncatedSVD model

        svd_2 = TruncatedSVD(n_components=2) 
        title = "Two TruncatedSVD projection directions (%s)"%dataset_name
        dimensionality_reduction.plot_2d(svd_2, title, xlabel, ylabel)
    
        svd_3 = TruncatedSVD(n_components=3) 
        title = "Three TruncatedSVD projection directions (%s)"%dataset_name
        dimensionality_reduction.plot_3d(svd_3, title, xlabel, ylabel, zlabel)
    
        # For PCA, what is the distribution of eigenvalues? For ICA, how kurtotic 
        # are the distributions? Do the projection axes for ICA seem to capture 
        # anything "meaningful"? 
        # distribution of PCA eigenvalues
        title = "Distribution of PCA eigenvalues (%s)"%dataset_name
        xlabel = "Number of components"
        if dataset_name == "voice dataset":
            ylabel = "Log(Eigenvalues)"
        else:
             ylabel = "Eigenvalues"
        x_axis = range(1, n_features + 1)
        dimensionality_reduction.eig_plot(PCA(n_features), x_axis, title, 
                                          xlabel, ylabel, dataset_name)
        
        title = "Distribution of TruncatedSVD eigenvalues (%s)"%dataset_name
        x_axis_svd = range(1, n_features)
        dimensionality_reduction.eig_plot(TruncatedSVD(n_features-1), x_axis_svd,
                                         title, xlabel, ylabel, dataset_name)        
        
        
        title = "Distribution of ICA kurtotic scores (%s)"%dataset_name
        ylabel = "Kurtotic scores"
        dimensionality_reduction.kurt_plot(x_axis, title, xlabel, ylabel)
    
        # Assuming you only generate k projections (i.e., you do dimensionality reduction), 
        # how well is the data reconstructed by the randomized projections? PCA? 
        # How much variation did you get when you re-ran your RP several times 
        # (I know I don't have to mention that you might want to run RP many times 
        # to see what happens, but I hope you forgive me)?

        # PCA reconstruction error
        title = "PCA reconstruction error (mean squared error) curve (%s)"%dataset_name
        xlabel = "Number of components"
        ylabel = "Mean squared error" 
        est = "PCA"
        dimensionality_reduction.rec_plot(x_axis, title, xlabel, ylabel, est, 
                                            is_fixed = False)
        # ICA reconstruction error
        title = "ICA reconstruction error (mean squared error) curve (%s)"%dataset_name
        est = "ICA"
        dimensionality_reduction.rec_plot(x_axis, title, xlabel, ylabel, est, 
                                            is_fixed = False)
                                            
        # TruncatedSVD reconstruction error                                        
        title = "TruncatedSVD reconstruction error (mean squared error) curve (%s)"%dataset_name
        est = "TruncatedSVD"
        dimensionality_reduction.rec_plot(x_axis_svd, title, xlabel, ylabel, est, 
                                          is_fixed = False)
        # RP reconstruction error
        title = "RP reconstruction error (mean squared error) curve (%s)"%dataset_name
        est = "RP"
        dimensionality_reduction.rec_plot(x_axis, title, xlabel, ylabel, est, 
                                            is_fixed = False)
        # RP reconstruction error with fixed projections 
        title = "RP reconstruction error curve with fixed three projections (%s)"%dataset_name
        est = "RP"
        dimensionality_reduction.rec_plot(x_axis, title, xlabel, ylabel, est, 
                                          is_fixed = True)
        # compare clustering with or without dimensionality reduction
        # When you reproduced your clustering experiments on the datasets projected 
        # onto the new spaces created by ICA, PCA and RP, did you get the same 
        # clusters as before? Different clusters? Why? Why not?
    
        # test the performance of KMeans and EM with dimensionality reduction
        dr_names = ["PCA", "ICA", "RP", "TruncatedSVD"]   
        for dr_name in dr_names:
            if dr_name == "PCA":
                if dataset_name == "voice dataset":
                    dr_X = PCA(2).fit_transform(X)
                else:
                    dr_X = PCA(17).fit_transform(X)
                # add another dataset
                    
                cluster_dr = clustering(dr_X,y)
                # KMeans and PCA
                xlabel = "Number of clusters"
                ylabel = "Score value"
                title = "KMeans internal measure scores with PCA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, ylabel)
                title = "KMeans external measure scores with PCA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, is_inertia = False)
                # EM and PCA
                title = "EM internal measure scores with PCA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, 
                                         ylabel, alg = "em")
                title = "EM external measure scores with PCA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, 
                                         is_inertia = False, alg = "em")
            elif dr_name == "ICA":
                if dataset_name == "voice dataset":
                    dr_X = FastICA(n_features).fit_transform(X)
                else:
                    dr_X = FastICA(61).fit_transform(X)
                # add another dataset
                cluster_dr = clustering(dr_X,y)
                # Kmeans and ICA
                xlabel = "Number of clusters"
                ylabel = "Score value"
                title = "KMeans internal measure scores with ICA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, ylabel)
                title = "KMeans external measure scores with ICA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, is_inertia = False)
                # EM and ICA
                title = "EM internal measure scores with ICA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, 
                                         ylabel, alg = "em")
                title = "EM external measure scores with ICA (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, 
                                         is_inertia = False, alg = "em")
            elif dr_name == "RP":
                dr_X = RP.GaussianRandomProjection(3).fit_transform(X)
                cluster_dr = clustering(dr_X,y)
                # Kmeans and RP
                xlabel = "Number of clusters"
                ylabel = "Score value"
                title = "KMeans internal measure scores with RP (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, ylabel)
                title = "KMeans external measure scores with RP (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, is_inertia = False)
                # EM and RP
                title = "EM internal measure scores with RP (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, 
                                         ylabel, alg = "em")
                title = "EM external measure scores with RP (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, 
                                         is_inertia = False, alg = "em")
        
            elif dr_name == "TruncatedSVD":
                if dataset_name == "voice dataset":
                    dr_X = TruncatedSVD(2).fit_transform(X)
                else:
                    dr_X = TruncatedSVD(17).fit_transform(X)
                cluster_dr = clustering(dr_X,y)
                # Kmeans and TruncatedSVD
                xlabel = "Number of clusters"
                ylabel = "Score value"
                title = "KMeans internal measure scores with TruncatedSVD (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, ylabel)
                title = "KMeans external measure scores with TruncatedSVD (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, is_inertia = False)
                # EM and TruncatedSVD            
                title = "EM internal measure scores with TruncatedSVD (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs_inertia, title, xlabel, 
                                         ylabel, alg = "em")
                title = "EM external measure scores with TruncatedSVD (%s)"%dataset_name
                cluster_dr.final_plots(score_funcs, title, xlabel, ylabel, 
                                         is_inertia = False, alg = "em")
            else:
                print "Incorrect dimensionality reduction algorithm"
                
            # compare the clusters generated with KMeans and EM with and without
            # dimensionality reduction
            
            for dr_name in dr_names:
                new_cluster = clustering(X,y)
                if dr_name == "TruncatedSVD":
                    x_axis = range(1, n_features)
                    n_features = n_features -1
                for score_func in score_funcs[:1]:
                    # Kmeans
                    plt.figure(figsize = (8,8))
                    title = "Compare KMeans clusters with and without %s dimensionality reduction (%s)"%(dr_name, dataset_name)
                    xlabel = "Number of components"
                    ylabel = "Score value (%s)"%score_func.__name__            
                    scores = dimensionality_reduction.clustering_plot(score_func, 
                                est = "k_means", dr = dr_name, 
                                n_features = n_features, dataset_name = dataset_name)
                    new_cluster.make_plots(scores, score_func, x_axis, title, xlabel, ylabel)
                    plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
            
                    # EM
                    plt.figure(figsize = (8,8))
                    title = "Compare EM clusters with and without %s dimensionality reduction (%s)"%(dr_name, dataset_name)
                    xlabel = "Number of components"
                    ylabel = "Score value (%s)"%score_func.__name__            
                    scores = dimensionality_reduction.clustering_plot(score_func, 
                                est = "em", dr = dr_name, n_features = n_features, 
                                dataset_name = dataset_name)
                    new_cluster.make_plots(scores, score_func, x_axis, title, xlabel, ylabel)
                    plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
      
    
       
    # load dataset
    # voice dataset
    df1 = pd.read_csv("../Dataset/voice.csv")
    df1["label"].replace("male", 0, inplace = True)
    df1["label"].replace("female", 1, inplace = True)
    y_1 = df1["label"]
    df1.drop("label", inplace = True, axis = 1)
    X_1 = df1.values
    n_features1 = df1.shape[1]
    dataset_name1 = "voice dataset"
    
    # digit dataset
    digits = load_digits()
    X_2 = scale(digits.data)
    y_2 = digits.target
    n_features2 = X_2.shape[1]
    dataset_name2 = "digits dataset"

    # set up metrices
    score_funcs = [
    metrics.adjusted_rand_score,
    metrics.adjusted_mutual_info_score,
    metrics.homogeneity_score,
    metrics.completeness_score
    ]
    score_funcs_inertia = [metrics.silhouette_score]
    
    
    # analyze voice dataset (questions 1-3)
    analysis(X_1, y_1, score_funcs, score_funcs_inertia, n_features1, dataset_name1)
    # analyze EEG dataset (questions 1-3)    
    analysis(X_2, y_2, score_funcs, score_funcs_inertia, n_features2, dataset_name2)
    """
    # Apply the dimensionality reduction algorithms to one of your datasets from
    # assignment #1 (if you've reused the datasets from assignment #1 to do 
    # experiments 1-3 above then you've already done this) and rerun your neural 
    # network learner on the newly projected data.
    
    # setup optimized ann classifier and split dataset
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1,y_1, test_size=0.3, 
                                                        random_state=42)
    ann =  MLPClassifier(activation = "logistic", 
                         solver = "lbfgs", hidden_layer_sizes = (3, 60))

    # compare different dimensionalities
 
    dr_ests = [("PCA",PCA), ("ICA", FastICA), ("RP", RP.GaussianRandomProjection),
               ("TruncatedSVD", TruncatedSVD)]
    
    title = "Performance of Artificial Neural Network with dimensionality\n reduction algorithms (voice data)"
    xlabel = "Number of components"
    ylabel = "Accuracy score"
    colors = ["g", "b", "c", "m"]
    
    nna = neural_network_analysis(X_train_1, y_train_1)
    nna.performance(ann, dr_ests, X_1, y_1, X_train_1, y_train_1, X_test_1, y_test_1, title,
                        xlabel, ylabel, colors, n_features1)
    
    # Apply the clustering algorithms to the same dataset to which you just 
    # applied the dimensionality reduction algorithms (you've probably already 
    # done this), treating the clusters as if they were new features. In other 
    # words, treat the clustering algorithms as if they were dimensionality 
    # reduction algorithms. Again, rerun your neural network learner on the 
    # newly projected data.
    
    # compare different clustering algorithms
    dr_ests = [("KMeans",KMeans), ("EM", GaussianMixture)]
    title = "Performance of Artificial Neural Network with clustering\n algorithms (voice data)"
    xlabel = "Number of clusters"                    
    nna.performance(ann, dr_ests, X_1, y_1, X_train_1, y_train_1, X_test_1, y_test_1, title,
                        xlabel, ylabel, colors, n_features1, is_dr = False)
    
    # library versions
    print "sklearn version: ", sklearn.__version__
    print "pandas version: ", pandas.__version__
    print "numpy version: ", numpy.__version__
    print "matplotlib version: ", matplotlib.__version__
    print "scipy version: ", scipy.__version__
    """