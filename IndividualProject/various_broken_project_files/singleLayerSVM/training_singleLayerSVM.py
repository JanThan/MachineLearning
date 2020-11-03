# -*- coding: utf-8 -*-
"""
single layer SVM on MNIST
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

def show_data_example(i, dataset, label):
  # show some of the images in the dataset
  # call multiple times for multiple images
  # squeeze is necessary here to get rid of the extra dimension introduced in rehsaping
  print('\nExample Image: %s from selected dataset' %i)
  plt.imshow(np.squeeze(dataset[i]), cmap=plt.get_cmap('gray'))
  plt.show()
  print('label for image:', label[i,:])
  return

def flatten(Array):
  m,x,y,z = Array.shape
  n = x*y*z
  Array_flat = np.reshape(Array, (m,n))
  return Array_flat

def train(X, y, number_of_classes, save_model):
  # training final SVM
  print('training SVM classifier')
  SVMclassifier = []
  for i in range(number_of_classes):
    print('class number:', i )
    SVMclassifier.append(SVC(C=0.1, kernel='rbf', probability=True, decision_function_shape='ovr'))
    SVMclassifier[i].fit(X, y[:,i])
  print('training complete')
  if save_model == 'True':
    # save final SVM  
    print('saving model')
    joblib.dump(SVMclassifier, 'SVMclassifier.pkl', compress = 3)  
  return SVMclassifier

def get_accuracies(X_train, X_test, y_train, y_test, SVM):
  accuracy_train = []
  accuracy_test = []
  
  print('SVM classifier saved...\ngetting accuracies...')
  
  for i in range(y_train.shape[1]):
    print('model 1, number:',i)
    acctr = SVM[i].score(X_train, y_train[:,i])
    accte = SVM[i].score(X_test, y_test[:,i])
    accuracy_train.append(acctr)
    accuracy_test.append(accte)
    
  # plotting individual SVMs
  print('plotting accuracies')
  x = np.arange(10)
  w = 0.3
  # training accuracy
  print('training accuracy')
  plt.bar(x, accuracy_train, width=w, color='g')
  plt.ylim(0.95,1)
  plt.show()
  plt.savefig('singleLayerSVM_Training_accuracy')
  plt.close()
      
  # testing accuracy
  print('testing accuracy')
  plt.bar(x, accuracy_test, width=w, color='g')
  plt.ylim(0.95,1)
  plt.show()
  plt.savefig('singleLayerSVM_Testing_accuracy')
  plt.close()
  return

def main():
  print('loading presaved packets and feature data')
  packet = joblib.load('P:\Implementations\CNN_keras\MNIST\postGateway\data_packet.pkl')
  
  learning_rate, momentum, dropout, batch_size, epochs, decay,\
  number_of_classes, image_shape, \
  X_train, y_train, X_test, y_test = packet
  
  print('flattening X data')
  X_train_flat = flatten(X_train)
  X_test_flat = flatten(X_test)
  
  SVMclassifier = train(X_train_flat, y_train, number_of_classes, save_model='True')
  get_accuracies(X_train_flat, X_test_flat, y_train, y_test, SVMclassifier)

if __name__ == "__main__":
    main()
