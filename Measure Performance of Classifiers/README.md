## Performance of Classifiers for Gender Recognition and Brain Confusion Status Prediction

### ABSTRACT

Data are increasing and changing our world at an astonishing pace in almost every single science
field as well as in our life. With increasing amounts of data, it’s necessary to have fast, accurate
and robust algorithms for data modeling and prediction, such as advanced machine learning
algorithms. In general, machine learning algorithms may provide promising solution to help
people digging into data to find valuable information. In particular, for individual case study, it is
import to find the optimized algorithm for a certain data problem. To understand the
performance of different machine algorithms on different datasets and to find the optimized
algorithm to solve a data problem, this report aims to analyze the performance of different
classifiers on two interesting problems: gender recognition by voice and brain confusion status
prediction by electroencephalogram signals. Specifically, the following classifiers are analyzed:
Decision Trees, Artificial Neural Networks, Adaboost Classifier, Support Vector Machines and K
Nearest Neighbors. The variance-bias trade-off, under-fitting, over-fitting and the performance
of particular parameter of each algorithm are examined in two different datasets. Meanwhile,
several key parameters of each classifier are optimized and applied to predict the testing data.
The prediction performances on the testing data are measured with accuracy score. The final
performance of the optimized classifiers and the running time of each classifier are compared.
This assay thus provides an example to analyze the performance of different classifiers on
particular data problems.


### CODE INSTRUCTION

To run the code, please use Spyder IDE (version 2.3.8 for mac OS, https://github.com/spyder-ide/spyder/releases/tag/v2.3.8) with attached custom sklearn library (merged version of 0.17, 0.18 dev, 0.19 dev versions) to replace the sklearn library  in the Spyder IDE folder (Spyder-Py2.app/Contents/Resources/lib/python2.7). Please keep all subfolders (including Code folder, Dataset folder and Figures folder) in the zxie66 folder to run the program.

### DOCUMENT INSTRUCTION

1. Dataset folder includes two datasets for this analysis - voice.csv and EEG data.csv

2. Code folder contains two python documents: 

1) DataAnalysis.py - this document includes code to generate two figures for feature analysis for two datasets, and to compute the basic information of datasets.

2) ClassifierAnalysis.py - this document includes code to generate learning curves and validation curves, and to search the optimize parameters for each classifier.

The python code was written with the Spyder IDE (version 2.3.8) on mac OS X El Capitan (version 10.11.4). The version of python is Python 2.7.10. The following libraries were used to support the code: a). pandas (version 0.17.1), b). seaborn (version 0.6.0), c). matplotlib (version1.5.0), d). numpy (version 1.10.1), e). sklearn (version 0.17, 0.18 dev for tree class, 0.19 dev for neural network, svm and neighbors classes), the merged sklearn was included in the zxie66 folder.

NOTE for sklearn: because some new features such as tree pruning are not included in the old 0.17 version, the custom sklearn library was created. The latest dev 0.18 and 0.19 versions of sklearn was downloaded from https://github.com/scikit-learn/scikit-learn. To install the new class, please follow the following steps with mac terminal:  1). install pip on mac using command sudo easy_install pip, 2). install cython by running csh sudo pip install cython, 3). direct to the class folder and convert .pyx to .c by running cython -a xx.pyx, 4). direct to the class folder run python setup.py install 

