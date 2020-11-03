# -*- coding: utf-8 -*-
"""
This file is for training the subSVMs
feature1 and subSVM1 are the 1st of the 'last 3 layers' -> con64 layer prior to pooling
feature2 and subSVM2 is the middle of the other 2 layers -> fully connected layer after flattening
feature3 and subSVM3 are the last of the 'last 3 layers' -> layer prior to final classification


"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.externals import joblib


def get_features(cnn_model, X_train, y_train, X_test, y_test):
  feature_network1 = Model(cnn_model.input, cnn_model.get_layer('feature1').output)
  feature1_train = feature_network1.predict(X_train)
  feature1_test = feature_network1.predict(X_test)
  
  feature_network2 = Model(cnn_model.input, cnn_model.get_layer('feature2').output)
  feature2_train = feature_network2.predict(X_train)
  feature2_test = feature_network2.predict(X_test)
      
  feature_network3 = Model(cnn_model.input, cnn_model.get_layer('feature3').output)
  feature3_train = feature_network3.predict(X_train)
  feature3_test = feature_network3.predict(X_test)  
    
  # feature1_train and feature1_test have shape (m x 14 x 14 x 64)
  # to be accepted by SVM, reshaping to size: m x 12544
  m,x,y,z = feature1_train.shape
  n = x*y*z
  feature1_train = np.reshape(feature1_train, (m,n))
  feature1_test = np.reshape(feature1_test, (feature1_test.shape[0],n))
  
  features = feature1_train, feature1_test, feature2_train, feature2_test, feature3_train, feature3_test  
  return features # feature responses from CNN for all of the samples

def train_subSVMs(y_train, y_test, features, c=0.1, get_accuracies='True', save_model='True'):
  # train subSVMs on the feature responses
  print('\nTraining subSVMs')
  feature1_train, feature1_test, \
  feature2_train, feature2_test, \
  feature3_train, feature3_test = features
  subSVM1 = []
  subSVM2 = []
  subSVM3 = []
  print(feature1_train.shape, feature1_test.shape)
  for i in range(y_train.shape[1]):
    print('model 1, number:', i)
    subSVM1.append(SVC(C=c, kernel='rbf', probability=True, decision_function_shape='ovr'))
    subSVM1[i].fit(feature1_train, y_train[:,i])
    print('model 2, number:',i)
    subSVM2.append(SVC(C=c, kernel='rbf', probability=True, decision_function_shape='ovr'))
    subSVM2[i].fit(feature2_train, y_train[:,i])
    print('model 3, number:',i)
    subSVM3.append(SVC(C=c, kernel='rbf', probability=True, decision_function_shape='ovr'))
    subSVM3[i].fit(feature3_train, y_train[:,i])
  print('subSVMs trained...')
  
  if save_model == 'True':
    print('\nSaving subSVM models...')
    joblib.dump(subSVM1, 'saved_subSVM1.pkl', compress = 3)
    joblib.dump(subSVM2, 'saved_subSVM2.pkl', compress = 3)
    joblib.dump(subSVM3, 'saved_subSVM3.pkl', compress = 3)
    print('subSVM models saved')
    
  if get_accuracies=='True':
    print('\nFinding accuracies of trained subSVMS')
    accuracy1_test = [] 
    accuracy1_train = []
    accuracy2_test = [] 
    accuracy2_train = []
    accuracy3_test = [] 
    accuracy3_train = []
    
    for i in range(y_train.shape[1]):
      print('model 1, number:',i)
      acctr = subSVM1[i].score(feature1_train, y_train[:,i])
      accte = subSVM1[i].score(feature1_test, y_test[:,i])
      accuracy1_train.append(acctr)
      accuracy1_test.append(accte)
      
      print('model 2, number:', i)
      acctr = subSVM2[i].score(feature2_train, y_train[:,i])
      accte = subSVM2[i].score(feature2_test, y_test[:,i])
      accuracy2_train.append(acctr)
      accuracy2_test.append(accte)            
      
      print('model 3, number:', i)
      acctr = subSVM3[i].score(feature3_train, y_train[:,i])
      accte = subSVM3[i].score(feature3_test, y_test[:,i])
      accuracy3_train.append(acctr)
      accuracy3_test.append(accte)        
      
#      # predicted train and test values for a certain number, i 
#      yhat1_train = subSVM1[i].predict(feature1_train)
#      print(yhat1_train.shape)
#      print(yhat1_train)
#      yhat1_test = subSVM1[i].predict(feature1_test)
#      # test and train accuracies for that number
#      accte = np.mean(yhat1_test == y_test[:,i])
#      acctr = np.mean(yhat1_train == y_train[:,i])
#      accuracy1_test.append(accte)
#      accuracy1_train.append(acctr)
#      print('model 2, number:',i)
#      yhat2_train = subSVM2[i].predict(feature2_train)
#      yhat2_test = subSVM2[i].predict(feature2_test)
#      accte = np.mean(yhat2_test == y_test[:,i])
#      acctr = np.mean(yhat2_train == y_train[:,i])
#      accuracy2_test.append(accte)
#      accuracy2_train.append(acctr)
#      print('model 3, number:',i)
#      yhat3_train = subSVM3[i].predict(feature3_train)
#      yhat3_test = subSVM3[i].predict(feature3_test)
#      accte = np.mean(yhat3_test == y_test[:,i])
#      acctr = np.mean(yhat3_train == y_train[:,i])
#      accuracy3_test.append(accte)
#      accuracy3_train.append(acctr)  
      
  
    # plotting individual SVMs
    x = np.arange(10)
    w = 0.3
    # training accuracy
    print('training accuracy')
    plt.bar(x, accuracy1_train, width=w, color='g')
    plt.bar(x+w, accuracy2_train, width=w, color='b')
    plt.bar(x+2*w, accuracy3_train, width=w, color='r')
    plt.ylim(0.95,1)
    plt.show()
    plt.savefig('subSVM_Training_accuracy')
    plt.close()
        
    # testing accuracy
    print('testing accuracy')
    plt.bar(x, accuracy1_test, width=w, color='g')
    plt.bar(x+w, accuracy2_test, width=w, color='b')
    plt.bar(x+2*w, accuracy3_test, width=w, color='r')
    plt.ylim(0.95,1)
    plt.show()
    plt.savefig('subSVM_Testing_accuracy')
    plt.close()
    
  return subSVM1, subSVM2, subSVM3

def train(CNN_model, data, c, save_model = 'True', get_accuracies= 'True'):
  X_train, y_train, X_test, y_test = data
  features = get_features(CNN_model, X_train, y_train, X_test, y_test)
  subSVM1, subSVM2, subSVM3 = train_subSVMs(y_train, y_test, features, c, get_accuracies, save_model)
  return subSVM1, subSVM2, subSVM3, features 
      
  

