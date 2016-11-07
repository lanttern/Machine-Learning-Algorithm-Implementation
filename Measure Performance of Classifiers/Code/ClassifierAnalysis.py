# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:59:43 2016
Analyze the following classifiers:
1 - Decision trees with some form of pruning
2 - Neural networks
3 - Boosting
4 - Support Vector Machines
5 - k-nearest neighbors
@author: zhihuixie
"""
import sklearn
from sklearn import tree
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn import cross_validation, grid_search, metrics
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

class ClassifierAnalysis():
    """
    This class performs analysis to evaluate the performance of 5 different 
    supervise learning classifiers
    """
    def __init__(self, X, y):
        """
        X variable as feature data, y variable labels
        """
        self.X = X
        self.y = y
        
    def decision_tree(self, min_impurity_splits = None, is_voice_data = True):
        """
        analyze decision tree algorithm with or without pruning 
        
        
        Parameters for DecisionTreeClassifier
        ----------
        criterion : string, optional (default="gini")
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.
        splitter : string, optional (default="best")
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.
        max_features : int, float, string or None, optional (default=None)
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.
        max_depth : int or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node:
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        min_weight_fraction_leaf : float, optional (default=0.)
            The minimum weighted fraction of the input samples required to be at a
            leaf node.
        max_leaf_nodes : int or None, optional (default=None)
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
        class_weight : dict, list of dicts, "balanced" or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``
            For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        min_impurity_split : float, optional (default=1e-7)
            Threshold for early stopping in tree growth. A node will split
            if its impurity is above the threshold, otherwise it is a leaf.
            .. versionadded:: 0.18
        presort : bool, optional (default=False)
            Whether to presort the data to speed up the finding of best splits in
            fitting. For the default settings of a decision tree on large
            datasets, setting this to true may slow down the training process.
            When using either a smaller dataset or a restricted depth, this may
            speed up the training.
        """
        title = "Learning Curves (Decision Tree - voice dataset)"
        if not is_voice_data:
            title = "Learning Curves (Decision Tree - EEG dataset)"
        estimators = []
        for min_impurity_split in min_impurity_splits:
            estimator = tree.DecisionTreeClassifier(criterion="entropy", \
                                      min_impurity_split = min_impurity_split)
            estimators.append(estimator)

        # set colors: r -red, g- green, b - blue, m - magenta
        colors = [("r", "g"), ("b", "m")] 
        labels = [("Training accuracy (unpruned tree)", 
                     "Cross-validation accuracy (unpruned tree)"),
                     ("Training accuracy (pruned tree)", 
                     "Cross-validation accuracy (pruned tree)")]
        
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 30% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=100,
                                  test_size=0.3, random_state=42)
        self.plot_learning_curve(estimators, title, labels, colors, self.X, self.y, \
                                             cv=cv, n_jobs=4)
        
        # plot validation curve
        estimator_val = tree.DecisionTreeClassifier (criterion="entropy") 
        param_name = "min_impurity_split"
        x_label = "Number of nodes in decision tree"
        val_title = "Validation Curve with Decision Tree (voice dataset)"
        params =[i/100.0 for i in range(1,50)]
        if not is_voice_data:
            val_title = "Validation Curve with Decision Tree (EEG dataset)"
            params = np.logspace(-0.25, 0, 50)
        number_of_nodes = []
        for param in params:
            clf = tree.DecisionTreeClassifier(criterion="entropy", min_impurity_split = param)
            clf.fit(self.X, self.y)
            number_of_nodes.append(clf.tree_.node_count)
        print number_of_nodes
        self.plot_validation_curve(estimator_val, params, param_name, self.X, 
               self.y, val_title, xtricks = number_of_nodes, x_label = x_label,
                cv=cv, n_jobs = 4)
        plt.show()
    
    def ann(self, layer_sizes, is_voice_data = True):
        """
        analyze neural network algorithm with different layers
        Parameters for MLPClassifier:	
            hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
                The ith element represents the number of neurons in the ith hidden layer.
            activation : {‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
                Activation function for the hidden layer.
                ‘identity’, no-op activation, useful to implement linear 
                bottleneck, returns f(x) = x
                ‘logistic’, the logistic sigmoid function, returns 
                f(x) = 1 / (1 + exp(-x)).
                ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
                ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
            algorithm : {‘l-bfgs’, ‘sgd’, ‘adam’}, default ‘adam’
                The solver for weight optimization.
                ‘lbgfs’ is an optimizer in the family of quasi-Newton methods.
                ‘sgd’ refers to stochastic gradient descent.
                ‘adam’ refers to a stochastic gradient-based optimizer proposed 
                by Kingma, Diederik, and Jimmy Ba
                Note: The default solver ‘adam’ works pretty well on relatively 
                large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbgfs’ can converge faster and perform better.
            alpha : float, optional, default 0.0001
            L2 penalty (regularization term) parameter.
            batch_size : int, optional, default ‘auto’
            Size of minibatches for stochastic optimizers. If the solver is 
            ‘lbgfs’, the classifier will not use minibatch. When set to “auto”, 
            batch_size=min(200, n_samples)
            learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
                Learning rate schedule for weight updates.
                ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
                ‘invscaling’ gradually decreases the learning rate learning_rate_ at 
                each time step ‘t’ using an inverse scaling exponent of ‘power_t’. 
                effective_learning_rate = learning_rate_init / pow(t, power_t)
                ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’
                as long as training loss keeps decreasing. Each time two consecutive 
                epochs fail to decrease training loss by at least tol, or fail to 
                increase validation score by at least tol if ‘early_stopping’ is on, 
                the current learning rate is divided by 5.
                Only used when solver='sgd'.
            max_iter : int, optional, default 200
                Maximum number of iterations. The solver iterates until convergence 
                (determined by ‘tol’) or this number of iterations.
            random_state : int or RandomState, optional, default None
                State or seed for random number generator.
            shuffle : bool, optional, default True
                Whether to shuffle samples in each iteration. Only used when 
                solver=’sgd’ or ‘adam’.
            tol : float, optional, default 1e-4
                Tolerance for the optimization. When the loss or score is not 
                improving by at least tol for two consecutive iterations, unless 
                learning_rate is set to ‘adaptive’, convergence is considered to 
                be reached and training stops.
            learning_rate_init : double, optional, default 0.001
                The initial learning rate used. It controls the step-size in 
                updating the weights. Only used when solver=’sgd’ or ‘adam’.
            power_t : double, optional, default 0.5
                The exponent for inverse scaling learning rate. It is used in 
                updating effective learning rate when the learning_rate is set 
                to ‘invscaling’. Only used when solver=’sgd’.
            verbose : bool, optional, default False
                Whether to print progress messages to stdout.
            warm_start : bool, optional, default False
                When set to True, reuse the solution of the previous call to fit 
                as initialization, otherwise, just erase the previous solution.
            momentum : float, default 0.9
                Momentum for gradient descent update. Should be between 0 and 1. 
                Only used when solver=’sgd’.
            nesterovs_momentum : boolean, default True
                Whether to use Nesterov’s momentum. Only used when solver=’sgd’ 
                and momentum > 0.
            early_stopping : bool, default False
                Whether to use early stopping to terminate training when validation 
                score is not improving. If set to true, it will automatically set 
                aside 10% of training data as validation and terminate training 
                when validation score is not improving by at least tol for two 
                consecutive epochs. Only effective when solver=’sgd’ or ‘adam’
            validation_fraction : float, optional, default 0.1
                The proportion of training data to set aside as validation set 
                for early stopping. Must be between 0 and 1. Only used if 
                early_stopping is True
            beta_1 : float, optional, default 0.9
                Exponential decay rate for estimates of first moment vector in adam, 
                should be in [0, 1). Only used when solver=’adam’
            beta_2 : float, optional, default 0.999
                Exponential decay rate for estimates of second moment vector in adam,
                should be in [0, 1). Only used when solver=’adam’
            epsilon : float, optional, default 1e-8
                Value for numerical stability in adam. Only used when solver=’adam’
        """        
        title = "Learning Curves (Artificial Neural Network  - voice dataset)"
        if not is_voice_data:
            title = "Learning Curves (Artificial Neural Network  - EEG dataset)"
        estimators = []
        for layer_size in layer_sizes:
            estimator = MLPClassifier(activation = "logistic",
                                      algorithm = "l-bfgs", 
                                      hidden_layer_sizes=layer_size)
            estimators.append(estimator)

        # set colors: r -red, g- green, b - blue, m - magenta
        colors = [("r", "g"), ("b", "m")] 
        labels = [("Training accuracy (single layer)", 
                     "Cross-validation accuracy (single layer)"),
                     ("Training accuracy (multiple layers)", 
                     "Cross-validation accuracy (multiple layers)")]
        
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 30% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=100,
                                   test_size=0.3, random_state=42)
        self.plot_learning_curve(estimators, title, labels, colors, self.X, self.y, \
                                            cv=cv, n_jobs=4)
        
        # plot validation curve
        estimator_val = MLPClassifier(activation = "logistic")
        param_name = "hidden_layer_sizes"
        x_label = "Number of layers in neural network"
        val_title = "Validation Curve with Artificial Neural Network (voice dataset)"
        params = [(3,1)] + [(3, i) for i in range(5,51, 5)]
        if not is_voice_data:
            val_title = "Validation Curve with Artificial Neural Network (EEG dataset)"
            params = [(3,1)] + [(3, i) for i in range(5,51, 5)]
           # ylim = (0.0, 1.01)
        layers = [layer_size[1] for layer_size in params]
        self.plot_validation_curve(estimator_val, params, param_name, self.X, 
               self.y, val_title, xtricks = layers, x_label = x_label,
               cv=cv, n_jobs = 4)
        plt.show()
    
    def boosting(self, num_estimators, is_voice_data = True):
        """
        analyze decision tree with boosting of different estimators
        Parameters for AdaBoostClassifier:
            base_estimator : object, optional (default=DecisionTreeClassifier)
                The base estimator from which the boosted ensemble is built. 
                Support for sample weighting is required, as well as proper 
                classes_ and n_classes_ attributes.
            n_estimators : integer, optional (default=50)
                The maximum number of estimators at which boosting is terminated. 
                In case of perfect fit, the learning procedure is stopped early.
            learning_rate : float, optional (default=1.)
                Learning rate shrinks the contribution of each classifier by 
                learning_rate. There is a trade-off between learning_rate and n_estimators.
            algorithm : {‘SAMME’, ‘SAMME.R’}, optional (default=’SAMME.R’)
                If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. 
                base_estimator must support calculation of class probabilities. 
                If ‘SAMME’ then use the SAMME discrete boosting algorithm. 
                The SAMME.R algorithm typically converges faster than SAMME, achieving 
                a lower test error with fewer boosting iterations.
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator; 
                If RandomState instance, random_state is the random number generator; 
                If None, the random number generator is the RandomState instance used by np.random.
        """
        title = "Learning Curves (AdaBoostClassifier  - voice dataset)"
        if not is_voice_data:
            title = "Learning Curves (AdaBoostClassifier  - EEG dataset)"
        estimators = []
        for num_estimator in num_estimators:
            estimator = AdaBoostClassifier(n_estimators = num_estimator)
            estimators.append(estimator)

        # set colors: r -red, g- green, b - blue, m - magenta
        colors = [("r", "g"), ("b", "m")] 
        labels = [("Training accuracy (n_estimator = %d)"%num_estimators[0], 
                     "Cross-validation accuracy (n_estimator = %d)"%num_estimators[0]),
                     ("Training accuracy (n_estimator = %d)"%num_estimators[1], 
                     "Cross-validation accuracy (n_estimator = %d)"%num_estimators[1])]
        
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 30% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=100,
                                   test_size=0.3, random_state=42)
        self.plot_learning_curve(estimators, title, labels, colors, self.X, self.y, \
                                            cv=cv, n_jobs=4)
        
        # plot validation curve
        estimator_val = AdaBoostClassifier()
        param_name = "n_estimators"
        x_label = "Number of estimators for boosting"
        val_title = "Validation Curve with AdaBoostClassifier (voice dataset)"
        params = range(10, 201, 10)
        if not is_voice_data:
            val_title = "Validation Curve with AdaBoostClassifier (EEG dataset)"
        self.plot_validation_curve(estimator_val, params, param_name, self.X, 
               self.y, val_title, xtricks = params, x_label = x_label,
               cv=cv, n_jobs = 4)
        plt.show()
    
    def svm_classifier(self, kernels, is_voice_data = True):
        """
        analyze support vector machines algorithm with different kernel functions
        Parameters for SVC:		
            C : float, optional (default=1.0)
                Penalty parameter C of the error term.
            kernel : string, optional (default=’rbf’)
                Specifies the kernel type to be used in the algorithm. It must 
                be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or 
                a callable. If none is given, ‘rbf’ will be used. If a callable 
                is given it is used to pre-compute the kernel matrix from data 
                matrices; that matrix should be an array of shape (n_samples, n_samples).
            degree : int, optional (default=3)
                Degree of the polynomial kernel function (‘poly’). Ignored by 
                all other kernels.
            gamma : float, optional (default=’auto’)
                Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is
                ‘auto’ then 1/n_features will be used instead.
            coef0 : float, optional (default=0.0)
                Independent term in kernel function. It is only significant in 
                ‘poly’ and ‘sigmoid’.
            probability : boolean, optional (default=False)
                Whether to enable probability estimates. This must be enabled 
                prior to calling fit, and will slow down that method.
            shrinking : boolean, optional (default=True)
                Whether to use the shrinking heuristic.
            tol : float, optional (default=1e-3)
                Tolerance for stopping criterion.
            cache_size : float, optional
                Specify the size of the kernel cache (in MB).
            class_weight : {dict, ‘balanced’}, optional
                Set the parameter C of class i to class_weight[i]*C for SVC. 
                If not given, all classes are supposed to have weight one. 
                The “balanced” mode uses the values of y to automatically 
                adjust weights inversely proportional to class frequencies 
                in the input data as n_samples / (n_classes * np.bincount(y))
            verbose : bool, default: False
                Enable verbose output. Note that this setting takes advantage 
                of a per-process runtime setting in libsvm that, if enabled, 
                may not work properly in a multithreaded context.
            max_iter : int, optional (default=-1)
                Hard limit on iterations within solver, or -1 for no limit.
            decision_function_shape : ‘ovo’, ‘ovr’ or None, default=None
                Whether to return a one-vs-rest (‘ovr’) ecision function of shape 
                (n_samples, n_classes) as all other classifiers, or the original 
                one-vs-one (‘ovo’) decision function of libsvm which has shape 
                (n_samples, n_classes * (n_classes - 1) / 2). The default of
                None will currently behave as ‘ovo’ for backward compatibility 
                and raise a deprecation warning, but will change ‘ovr’ in 0.18.
                New in version 0.17: decision_function_shape=’ovr’ is recommended.
                Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.
            random_state : int seed, RandomState instance, or None (default)
                The seed of the pseudo random number generator to use when 
                shuffling the data for probability estimation.
        """
        title = "Learning Curves (SVM  - voice dataset)"
        if not is_voice_data:
            title = "Learning Curves (SVM  - EEG dataset)"
                        
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 30% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=100,
                                   test_size=0.3, random_state=42)
        for func in kernels:
            estimators = [SVC(kernel = func, max_iter = 4000)]
            labels = [("Training accuracy (kernel = %s)"%func, 
                     "Cross-validation accuracy (kernel = %s)"%func)]
            # set colors: r -red, g- green
            colors = [("r", "g")]
            title = title + "_" + func
            self.plot_learning_curve(estimators, title, labels, colors, self.X, self.y, \
                                            cv=cv, n_jobs=4)
            plt.show()
    
    def knn(self, k_list, is_voice_data  = True):
        """
        analyze k-nearest neighbors algorithm with differnt k-neighbors
        Parameters for KNeighborsClassifier:	
            n_neighbors : int, optional (default = 5)
                Number of neighbors to use by default for k_neighbors queries.
            weights : str or callable
                weight function used in prediction. Possible values:
                ‘uniform’ : uniform weights. All points in each neighborhood 
                            are weighted equally.
                ‘distance’ : weight points by the inverse of their distance. 
                            in this case, closer neighbors of a query point will 
                            have a greater influence than neighbors which are further away.
                [callable] : a user-defined function which accepts an array of 
                            distances, and returns an array of the same shape 
                            containing the weights.
                Uniform weights are used by default.
            algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
                Algorithm used to compute the nearest neighbors:
                    ‘ball_tree’ will use BallTree
                    ‘kd_tree’ will use KDTree
                    ‘brute’ will use a brute-force search.
                    ‘auto’ will attempt to decide the most appropriate algorithm 
                    based on the values passed to fit method.
            Note: fitting on sparse input will override the setting of this 
                parameter, using brute force.
            leaf_size : int, optional (default = 30)
                Leaf size passed to BallTree or KDTree. This can affect the speed 
                of the construction and query, as well as the memory required to 
                store the tree. The optimal value depends on the nature of the problem.
            metric : string or DistanceMetric object (default = ‘minkowski’)
                the distance metric to use for the tree. The default metric is 
                minkowski, and with p=2 is equivalent to the standard Euclidean 
                metric. See the documentation of the DistanceMetric class for a 
                list of available metrics.
            p : integer, optional (default = 2)
                Power parameter for the Minkowski metric. When p = 1, this is 
                equivalent to using manhattan_distance (l1), and euclidean_distance 
                (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
            metric_params : dict, optional (default = None)
                Additional keyword arguments for the metric function.
            n_jobs : int, optional (default = 1)
                The number of parallel jobs to run for neighbors search. If -1, 
                then the number of jobs is set to the number of CPU cores. 
                Doesn’t affect fit method.
        """
        title = "Learning Curves (KNeighborsClassifier  - voice dataset)"
        if not is_voice_data:
            title = "Learning Curves (KNeighborsClassifier  - EEG dataset)"
        estimators = []
        for k in k_list:
            estimator = KNeighborsClassifier(n_neighbors = k)
            estimators.append(estimator)

        # set colors: r -red, g- green, b - blue, m - magenta
        colors = [("r", "g"), ("b", "m")] 
        labels = [("Training accuracy (n_neighbor = %d)"%k_list[0], 
                     "Cross-validation accuracy (n_neighbor = %d)"%k_list[0]),
                     ("Training accuracy (n_neighbor = %d)"%k_list[1], 
                     "Cross-validation accuracy (n_neighbor = %d)"%k_list[1])]
        
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 30% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=100,
                                   test_size=0.3, random_state=42)
        self.plot_learning_curve(estimators, title, labels, colors, self.X, self.y, \
                                            cv=cv, n_jobs=4)
        
        # plot validation curve
        estimator_val = KNeighborsClassifier()
        param_name = "n_neighbors"
        x_label = "Number of neighbors"
        val_title = "Validation Curve with KNeighborsClassifier (voice dataset)"
        params = [1] + range(5, 101, 5)
        if not is_voice_data:
            val_title = "Validation Curve with KNeighborsClassifier (EEG dataset)"
            params = [1] + range(5, 201, 10)
        self.plot_validation_curve(estimator_val, params, param_name, self.X, 
               self.y, val_title, xtricks = params, x_label = x_label,
               cv=cv, n_jobs = 4)
        plt.show()
    
    def plot_learning_curve(self, estimators, title, labels, colors, X, y, 
            ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
        """
        Generate a simple plot of the test and traning learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : integer, cross-validation generator, optional
            If an integer is passed, it is the number of folds (defaults to 3).
            Specific cross-validation objects can be passed, see
            sklearn.cross_validation module for the list of possible objects

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure(figsize = (6,8))
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        for i in range(len(estimators)):      
            estimator = estimators[i]
            line_label = labels[i]
            line_color = colors[i]
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color=line_color[0])
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color=line_color[1])
            plt.plot(train_sizes, train_scores_mean, "o-", color=line_color[0],
                 label=line_label[0])#"Training accuracy"
            plt.plot(train_sizes, test_scores_mean, "o-", color=line_color[1],
                 label=line_label[1]) #"Cross-validation accuracy"

        plt.grid()        
        plt.legend(bbox_to_anchor=(0., 1.05, 1., .105),loc=3, mode="expand",
                   borderaxespad=0.)
        plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
        return plt
        
    def plot_validation_curve(self, estimator, params, param_name, X, y, title, 
              xtricks =None, x_label = None, ylim = None, cv=None, n_jobs = 1):
        
        plt.figure(figsize = (6,8))
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.title(title)#"Validation Curve with SVM"
        plt.xlabel(x_label)
        plt.ylabel("Accuracy")
        #param_range = np.logspace(-6, -1, 5)
        train_scores, test_scores = validation_curve(
                     estimator, X, y, param_name=param_name, param_range=params,
                     cv=cv, scoring="accuracy", n_jobs=n_jobs)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.plot(xtricks, train_scores_mean, "o-", label="Training accuracy", color="r")
        plt.fill_between(xtricks, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
        plt.plot(xtricks, test_scores_mean, "o-", label="Cross-validation accuracy",
             color="g")
        plt.fill_between(xtricks, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
        plt.grid()    
        plt.legend(loc="best")
        plt.savefig("../Figures/" + title + ".png", bbox_inches="tight")
        return plt
        
    def parameter_optimize(self, estimator, parameters, X_test, y_test):
        """
        To search the optimized parameters
        """
        cv = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=100,
                                   test_size=0.3, random_state=42)
        clf = grid_search.GridSearchCV(estimator, parameters[1], cv = cv, n_jobs =4)
        t1 = time.time()
        clf.fit(self.X, self.y)
        print "The optimize parameters for %s is: %s"%(parameters[0], clf.best_params_)
        y_pred = clf.predict(X_test)
        t2 = time.time()
        print "The running time for %s is: %f sec"%(parameters[0], t2 - t1)
        score = metrics.accuracy_score(y_test, y_pred)
        print "The accuracy score for %s is: %f"%(parameters[0], score), "\n"
        return {"%s"%parameters[0]: {"estimator_parameters": clf.best_params_, 
                "running_time": t2-t1, "accuracy_score": score}}
    def running_time_analysis(self, estimator, X_test, y_test):
        """
        this method print running time for optimized classifiers after training 
        and predicting test dataset
        """
        t1 = time.time()
        clf = estimator[1]
        clf.fit(self.X, self.y)
        y_pred = clf.predict(X_test)
        t2 = time.time()
        print "The running time for %s is: %f sec"%(estimator[0], t2 - t1)
        
    
if __name__ == "__main__":
    
    # learning curve and validation curve for dataset 1 - voice dataset
    df1 = pd.read_csv("../Dataset/voice.csv")
    df1["label"].replace("male", 0, inplace = True)
    df1["label"].replace("female", 1, inplace = True)
    y_1 = df1["label"]
    df1.drop("label", inplace = True, axis = 1)
    X_1 = df1.values
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1,y_1, test_size=0.3, 
                                                        random_state=42)
    
    ca1 = ClassifierAnalysis(X_train_1,y_train_1)
    
    ca1.decision_tree(min_impurity_splits = [0, 0.1])
    ca1.ann(layer_sizes = [(3,1), (3,10)])
    ca1.boosting(num_estimators = [5, 100])
    ca1.svm_classifier(kernels = ["linear", "rbf", "sigmoid"])
    ca1.knn(k_list = [1, 50])
    
    
    # learning curve and validation curve for dataset 2 - EEG dataset
    df2 = pd.read_csv("../Dataset/EEG data.csv")
    y_2 = df2["Self-defined label"]
    df2.drop("Self-defined label", inplace = True, axis = 1)
    df2.drop("predefined label", inplace = True, axis = 1)
    df2.drop("subject ID", inplace = True, axis = 1)
    df2.drop("Video ID", inplace = True, axis = 1)
    X_2 = df2.values
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2,y_2, test_size=0.3, 
                                                        random_state=42)
    
    ca2 = ClassifierAnalysis(X_train_2,y_train_2)
    
    ca2.decision_tree(min_impurity_splits = [0, 0.75], is_voice_data = False)
    ca2.ann(layer_sizes = [(3,1), (3,10)], is_voice_data = False) 
    ca2.boosting(num_estimators = [5, 100], is_voice_data = False)
    ca2.svm_classifier(kernels = ["linear", "rbf", "sigmoid"], is_voice_data = False) #, "rbf", "sigmoid"]
    ca2.knn(k_list = [1, 50], is_voice_data  = False)
    
    
    # search optimized parameters for each classifier
    estimators = [tree.DecisionTreeClassifier(), MLPClassifier(), \
            AdaBoostClassifier(), KNeighborsClassifier(), SVC(max_iter = 4000)]
    
              
    parameters_list1 = [("Decision Tree", {"criterion": ["gini","entropy"], 
                        "class_weight": ["balanced", None], 
                        "min_impurity_split": [i/100.0 for i in range(1,51)]}),
                      ("Neural Network", {"hidden_layer_sizes": [(3,1)] + [(3, i) for i in range(5,101, 5)],
                      "activation" : ["logistic", "tanh", "relu"],
                      "algorithm": ["l-bfgs", "sgd", "adam"]}),
                      ("Boosting", {"n_estimators":range(10, 201, 10), "learning_rate": np.logspace(-5, 0, 5),
                       "algorithm" : ["SAMME", "SAMME.R"]}),
                      ("KNN", {"n_neighbors":[1] + range(5, 101, 5), 
                       "algorithm" :["auto", "ball_tree", "kd_tree", "brute"],
                      "metric": ["euclidean", "manhattan", "minkowski"]}),
                      ("SVM", {"C":np.logspace(1, 5, 5), "kernel":["linear", "rbf", "sigmoid"],
                       "tol": np.logspace(-7, -2, 5)})]
    

    
    parameters_list2 = [("Decision Tree", {"criterion": ["gini","entropy"], 
                        "class_weight": ["balanced", None], 
                        "min_impurity_split": np.logspace(-0.25, 0, 50)}),
                      ("Neural Network", {"hidden_layer_sizes": [(3,1)] + [(3, i) for i in range(5,101, 5)],
                      "activation" : ["logistic", "tanh", "relu"],
                      "algorithm": ["l-bfgs", "sgd", "adam"]}),
                      ("Boosting", {"n_estimators":range(10, 201, 10), "learning_rate": np.logspace(-5, 0, 5),
                       "algorithm" : ["SAMME", "SAMME.R"]}),
                      ("KNN", {"n_neighbors":[1] + range(5, 201, 10), 
                       "algorithm" :["auto", "ball_tree", "kd_tree", "brute"],
                       "metric": ["euclidean", "manhattan", "minkowski"]}),
                       ("SVM", {"C":np.logspace(1, 5, 5), "kernel":["linear", "rbf", "sigmoid"],
                       "tol": np.logspace(-7, -2, 5)})]
    
    parameters_1 = []
    parameters_2 = []
    
    for (estimator, parameters) in zip(estimators, parameters_list1):
        d1 = ca1.parameter_optimize(estimator, parameters, X_test_1, y_test_1)
        parameters_1.append(d1)
    
    for (estimator, parameters) in zip(estimators, parameters_list2):    
        d2 = ca2.parameter_optimize(estimator, parameters, X_test_2, y_test_2)
        parameters_2.append(d2)
        
    print parameters_1, "\n" 
    print parameters_2
    
    
    # further test AdaBoostClassifier for more estimators
    ada_estimator = [AdaBoostClassifier (learning_rate = 0.056234132519034911, 
          algorithm = "SAMME.R")]
    ada_parameters = [("Boosting", {"n_estimators":range(200, 1201, 100)})]
    for (ada_estimator, ada_parameter) in zip(ada_estimator, ada_parameters):    
        ca2.parameter_optimize(ada_estimator, ada_parameter, X_test_2, y_test_2)
    
    # running time analysis for optimized classifier on predicting test data
    opt_estimators_voice = [("Decision Tree", tree.DecisionTreeClassifier(
           min_impurity_split = 0.1, criterion = "entropy", class_weight = None)),
           ("Neural Network", MLPClassifier(activation = "logistic", 
           algorithm = "l-bfgs", hidden_layer_sizes = (3, 60))),
          ("Boosting", AdaBoostClassifier (n_estimators = 180, learning_rate = 1.0, 
          algorithm = "SAMME.R")),
          ("SVM", SVC(max_iter = 4000, kernel = "rbf", 
          C = 100.0, tol =9.9999999999999995e-08)),
          ("KNN", KNeighborsClassifier(n_neighbors = 5, metric = "manhattan", algorithm = "auto"))]   
          
    opt_estimators_eeg = [("Decision Tree", tree.DecisionTreeClassifier(
           min_impurity_split = 0.92105531768948168, criterion = "entropy", class_weight = "balanced")),
           ("Neural Network", MLPClassifier(activation = "tanh", 
           algorithm = "adam", hidden_layer_sizes = (3, 80))),
          ("Boosting", AdaBoostClassifier (n_estimators = 900, learning_rate = 0.056234132519034911, 
          algorithm = "SAMME.R")),
          ("SVM", SVC(max_iter = 4000, kernel = "sigmoid", 
          C = 10.0, tol =9.9999999999999995e-08)),
          ("KNN", KNeighborsClassifier(n_neighbors = 75, 
          metric = "manhattan", algorithm = "auto"))]  
    for i in range(5):
        est1 = opt_estimators_voice[i]
        est2 = opt_estimators_eeg[i]
        ca1.running_time_analysis(est1, X_test_1, y_test_1)
        ca2.running_time_analysis(est2, X_test_2, y_test_2)
        
    # versions of packages
    print "The version of pandas: ", pd.__version__
    print "The version of numpy: ", np.__version__
    print "The version of matplotlib: ", matplotlib.__version__
    print "The version of sklearn: ", sklearn.__version__
    
    
    