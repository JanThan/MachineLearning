# -*- coding: utf-8 -*-
"""
Script for training the combined SVM using outputs from the subSVMs
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.svm import SVC

def get_subSVM_outs(features, data, subSVMs, number_of_classes):
  feature1_train, feature1_test,\
  feature2_train, feature2_test,\
  feature3_train, feature3_test = features
  
  X_train, y_train, X_test, y_test = data
  
  subSVM1, subSVM2, subSVM3 = subSVMs
  
  subSVM1_train_out = np.zeros((y_train.shape[0],number_of_classes))
  subSVM1_test_out = np.zeros((y_test.shape[0],number_of_classes))
  subSVM2_train_out = np.zeros((y_train.shape[0],number_of_classes))
  subSVM2_test_out = np.zeros((y_test.shape[0],number_of_classes))
  subSVM3_train_out = np.zeros((y_train.shape[0],number_of_classes))
  subSVM3_test_out = np.zeros((y_test.shape[0],number_of_classes))
  for i in range(number_of_classes):
    # predict_proba outputs 2 columns and m rows.
    # the 2 columns are as such: [prob of 0, prob of 1]
    # I will be taking the prob of 1 moving forward
    print(i)
    subSVM1_train_out[:,i] = subSVM1[i].predict_proba(feature1_train)[:,1]
    subSVM1_test_out[:,i] = subSVM1[i].predict_proba(feature1_test)[:,1]
    subSVM2_train_out[:,i] = subSVM2[i].predict_proba(feature2_train)[:,1]
    subSVM2_test_out[:,i] = subSVM2[i].predict_proba(feature2_test)[:,1]
    subSVM3_train_out[:,i] = subSVM3[i].predict_proba(feature3_train)[:,1]
    subSVM3_test_out[:,i] = subSVM3[i].predict_proba(feature3_test)[:,1]
    
  subSVM_concat_train_out = np.hstack([subSVM1_train_out, subSVM2_train_out, subSVM3_train_out])
  subSVM_concat_test_out = np.hstack([subSVM1_test_out, subSVM2_test_out, subSVM3_test_out])
  
  return subSVM_concat_train_out, subSVM_concat_test_out

def train(data, features, subSVMs, number_of_classes, get_accuracies='True'):
  subSVM_concat_train_out,\
  subSVM_concat_test_out = get_subSVM_outs(features, data, subSVMs, number_of_classes)
  
  X_train, y_train, X_test, y_test = data
  
  # training final SVM
  finalSVM = []
  for i in range(number_of_classes):
    finalSVM.append(SVC(C=0.1, kernel='rbf', probability=True, decision_function_shape='ovr'))
    finalSVM[i].fit(subSVM_concat_train_out, y_train[:,i])
  
  # save final SVM  
  print('\nSaving final SVM model...')
  joblib.dump(finalSVM, 'saved_finalSVM.pkl', compress = 3)
  print('Final SVM model saved')
  
  
  if get_accuracies=='True':
    print('\nFinding accuracies of trained subSVMS')
    accuracy_test = [] 
    accuracy_train = []
    
    for i in range(number_of_classes):
      print('final SVM model, number:',i)
      # predicted train and test values for a certain number, i 
      yhat1_train = finalSVM[i].predict(subSVM_concat_train_out)
      yhat1_test = finalSVM[i].predict(subSVM_concat_test_out)
      # test and train accuracies for that number
      accte = np.mean(yhat1_test == y_test[:,i])
      acctr = np.mean(yhat1_train == y_train[:,i])
      accuracy_test.append(accte)
      accuracy_train.append(acctr)
  
    # plotting individual SVMs
    x = np.arange(10)
    w = 0.3
    # training accuracy
    print('training accuracy')
    plt.bar(x, accuracy_train, width=w, color='g')
    plt.ylim(0.95,1)
    plt.show()
    plt.savefig('Final_SVM_Training_accuracy.png')
    plt.close()
    
    # testing accuracy
    print('testing accuracy')
    plt.bar(x, accuracy_test, width=w, color='g')
    plt.ylim(0.95,1)
    plt.show()
    plt.savefig('Final_SVM_Testing_accuracy.png')
    plt.close()
    
  return finalSVM
