## Performance of Clustering and Dimensionality Reduction Algorithms and Their Applications to Neural Network

### ABSTRACT

Nowadays, data are center in our life in many aspects. With increasing amounts of complex data having
many variables, it’s important and necessary to have algorithms to understand the patterns of these data
and to reduce the number of random variables to help this process. Because of this, the report here tries to
apply algorithms such as unsupervised clustering algorithms to cluster the data and apply dimensionality
reduction algorithms to do feature transformation. Specifically, two clustering algorithms, the K-means
and the expectation-maximization (EM) algorithms, were applied to two datasets (recognition of gender
by voice [voice dataset] and identify handwritten digits [digits dataset]). The performance of these
clustering algorithms was analyzed with internal and external metrics. In addition, four dimensionality
reduction algorithms (Principal Component Analysis [PCA], Independent Component Analysis [ICA],
Gaussian Random Projection [RP] and Truncated Singular Value Decomposition [TruncatedSVD]) were
tested on these two datasets. Next, the components/dimensions of each dimensionality reduction
algorithm showing best performance were selected for feature transformation and the transformed data
were applied for clustering algorithms to perform further analyses and comparisons. Finally, the
clustering or the dimensionality reduction algorithms transformed data were applied to neural network
classifier on voice dataset. The performance of different trained neural networks was compared and
analyzed. This assay provides a paradigm for analysis of the performance of clustering and
dimensionality reduction algorithms, as well as gives an example of application of these algorithms to
improve neural network classifier.


### CODE INSTRUCTION

To run the code, please use Spyder IDE (version 2.3.8 for mac OS, https://github.com/spyder-ide/spyder/releases/tag/v2.3.8) with attached custom sklearn library (modified v0.18) to replace the sklearn library  in the Spyder IDE folder (Spyder-Py2.app/Contents/Resources/lib/python2.7). 

### DOCUMENT INSTRUCTION

The folder includes the following subfolders or document:
1. Dataset folder includes two datasets for this analysis - voice.csv
2. Code folder contains one python document: 
CADRAnalysis.py - this document includes code to generate results and graphs for the analysis.
The python code was written with the Spyder IDE (version 2.3.8) on mac OS X El Capitan (version 10.11.4). The version of python is Python 2.7.10. The following libraries were used to support the code: a). pandas (version 0.17.1), b). matplotlib (version1.5.0), c). numpy (version 1.10.1), d). sklearn (version 0.18 with edits to random_projection.py class). NOTE: the random projection algorithm may generate some infinite values which could terminate the program. In that case, please re-run the program to make it works.
