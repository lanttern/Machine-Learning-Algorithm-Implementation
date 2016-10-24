## Randomized Optimization for Neural Network Prediction

### ABSTRACT

Machine learning algorithms are usually using parameters to control the learning rate or the
capacity of the underlying model. To obtain better prediction performance, it’s important to tune
the parameters to optimize the machine learning algorithms. The problem of optimization of a
machine learning algorithm, such as find the optimized parameters for the model, often can be
transformed to solve optimization problems using established optimization algorithms. Here, this
report focuses on applying randomized optimization algorithms, including Randomized Hill
Climbing (RHC), Simulated Annealing (SA), and Genetic Algorithm (GA), to solve the
optimization of a neural network problem. The performance of each optimization algorithm on
the optimization of the weights of the neural network are compared and discussed. To further
explore the performance of each optimization algorithm, the three algorithms mentioned above
together with another algorithm, the Mutual-Information-Maximizing Input Clustering (MIMIC),
are applied to solve three additional optimization problems – the knapsack problem, the n-queens
problem and the four peaks problem, are chosen. The performance of each algorithm on each
optimization problem are tested and analyzed through a serial of experiments. The parameters of
each algorithm are examined for each problem. The final performance of each algorithm on these
problems is compared and the best algorithm for each problem is highlighted. This report thus
explores the behaviors of different random search algorithms in solving neural network classifier
optimization and other classic optimization problems.

### CODE INSTRUCTION

To run the python code to split the dataset, please use Spyder IDE (version 2.3.8 for mac OS, https://github.com/spyder-ide/spyder/releases/tag/v2.3.8). The run the java code, please import the entire projects using Eclipse IDE (ars.2 Release (4.5.2)).

### DOCUMENT INSTRUCTION

1. Dataset folder includes 6 csv files (2 training files, 2 validation files and 2 testing files) generated from the EEG datasets for the analysis.

2. Code folder contains one python document and two java projects: 

1) split_data.py - this document splits the EEG dataset into training, validation and testing datasets.

2) RandomSearchANN java project- this project includes code to generate data for applying randomized optimization algorithms to the neural network classifier.

3) OptimizationProblems java project - this project includes code to generate data for applying randomized optimization algorithms to three classic optimization problems. 


