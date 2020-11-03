# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:46:21 2019

@author: janit
"""

# Load the libraries 
#import sys 
#import numpy # Mathematics functions, Arrays, Matrices
#import scipy # Optimisation, Linear algebra, signal/image processing

import matplotlib.pyplot as plt # Visualisation of data
import pandas # Data manipulation + Analysis
from pandas.plotting import scatter_matrix # draw a matrix of scatter plots
#import sklearn # General ML toolbox, Linear Regression, Classification etc.
from sklearn import model_selection
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# models to be used later
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset
file = "irisData.txt"
names = ['sepal-length','sepal-width','petal-length','petal-width','class'] 
dataset = pandas.read_csv(file, names=names) # dataset loaded

# DIMENSIONS  
print(dataset.shape) # X = 150x5... m = 150, i = 5

# Snippet of datatset
print(dataset.head(10)) # see first 10 rows of dataset

# Check basic stats about the dataset ... max/min/mean/count etc
print('\n', dataset.describe())

# Class Distribution
# check how much data is dedicated to each class 
print('\n', dataset.groupby('class').size()) # 50 for each class

# Data Visualisation
# Univariate (plot each variable) and Multivariate (interactions between variables) plots 

# ... Univariate
dataset.plot(kind='box', subplots = True, layout=(2,2), sharex=False, sharey=False, figsize=(10,10)) # box plot
#plt.show()
dataset.hist(figsize=(10,10)) # histogram
#plt.show()

# ... Multivariate
scatter_matrix(dataset, figsize=(10,10)) 
print('\n')

##############################################################################
# Splitting the dataset into Training(80%) and Validation (20%)
array = dataset.values
X = array[:,0:4] # First 4 columns are inputs 
Y = array[:,4] # 5th column --> Classes ... the output
validation_size = 0.2 # 20% validation set size
seed = 7 # used for random initialisation. Change this for different initialisations
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,
                                                    test_size=validation_size, random_state=seed)

# Test harness - k-Fold Cross Validation Technique
# Shuffle dataset randomly, split into 'k' groups, 
# train/model/test with each combination of the groups (train on k-1, test on 1)
# evaluate and retain a score of how each did (ie. accuracy)
seed = 7
scoring = 'accuracy' 

# Modeling
# SIMPLE LINEAR .... Logistic Regression, Linear Discriminant Analysis, 
# NON LINEAR .... K-nearest Neighbours, Classification and Regression Trees, 
                   # Gaussian Naive Bayes, Support Vector Machines

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results=[]
names=[]
for name, model in models:
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  string = " %s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
  print(string)

# visualise accuracy of methods
fig= plt.figure(figsize=(10,10))
fig.suptitle('Comparison')
plt.boxplot(results)
ax = fig.add_subplot()
ax.set_xticklabels(names)
plt.show()

# based on results, SVM used... any of listed methods would work given the result
KNN = SVC(gamma='auto')
KNN.fit(X_train, Y_train)
predictions = KNN.predict(X_validation)
# print(confusion_matrix(Y_validation, predictions))
print(accuracy_score(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# 90% accuracy with v.good f1, prescision and recall results