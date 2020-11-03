
# Imports ##
# general
import matplotlib.pyplot as plt
import numpy as np

# ML libraries
from tensorflow.keras.datasets import mnist

# ML utilities
from tensorflow.keras.utils import to_categorical

import os
os.chdir('P:\Implementations\CNN_keras\MNIST\postGateway')

# Python scripts used
import train_CNN
import load_CNN
import train_subSVMs
import load_subSVMs
import train_finalSVM
import load_finalSVM

import joblib

def save_array(array, name):
  joblib.dump(array, name+'.pkl', compress = 3)
  return
  
def load_array(array, name):
  array = joblib.load(array, name)
  return array

def show_data_example(i, dataset):
  # show some of the images in the dataset
  # call multiple times for multiple images
  # squeeze is necessary here to get rid of the extra dimension introduced in rehsaping
  print('\nExample Image: %s from selected dataset' %i)
  plt.imshow(np.squeeze(dataset[i]), cmap=plt.get_cmap('gray'))
  plt.show()
  return

def show_image(img):
  plt.imshow(img.squeeze(), cmap=plt.get_cmap('binary'))
  plt.show()  
  return

def load_and_encode(target_shape):
  # load dataset
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train, y_train = X_train[:,:,:],y_train[:]
  X_test, y_test =  X_test[:,:,:], y_test[:]
  print('Loaded Mnist dataset')
  print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
  print('Test:  X=%s, y=%s' % (X_test.shape, y_test.shape))
  # encode y data
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  # normalise X data (X/255 -> [0,1])
  X_train = X_train/255.0
  X_test = X_test/255.0
  # currently dimensions are (m x 28 x 28)
  # making them into (m x 28x28x1) 3Dimensional for convolution networks
  X_train = X_train.reshape(X_train.shape[0], target_shape[0], target_shape[1], target_shape[2])
  X_test = X_test.reshape(X_test.shape[0], target_shape[0], target_shape[1], target_shape[2])
    
  # show an arbitary example image from training set
  show_data_example(12, X_train)
  
  return X_train, y_train, X_test, y_test


image_shape = (28,28,1)
# load and encode mnist data
X_train, y_train, X_test, y_test = load_and_encode(image_shape)

# hyper-parameters
learning_rate = 0.1
momentum = 0.9
dropout = 0.5
batch_size = 128
epochs = 50
decay = 1e-6
number_of_classes = 10

# store required data into a packet to send to various imports
packet = [learning_rate, momentum, dropout, batch_size, epochs, decay,
          number_of_classes, image_shape,
          X_train, y_train, X_test, y_test]

data = [X_train, y_train, X_test, y_test]

CNN_model = load_CNN.load_model(packet)

subSVM1, subSVM2, subSVM3, features = load_subSVMs.load(CNN_model, data, c=0.1, get_accuracies='False')

subSVMs = [subSVM1, subSVM2, subSVM3]
feature1_train, feature1_test,\
feature2_train, feature2_test,\
feature3_train, feature3_test = features

final_SVM = joblib.load('saved_finalSVM.pkl')



"""
"""
from tensorflow.keras.models import Model


def predict_DNR(image):
  feature_network1 = Model(CNN_model.input, CNN_model.get_layer('feature1').output)
  feature_network2 = Model(CNN_model.input, CNN_model.get_layer('feature2').output)
  feature_network3 = Model(CNN_model.input, CNN_model.get_layer('feature3').output)
  
  output1 = np.expand_dims(feature_network1.predict(np.expand_dims(image,axis=0)).flatten(),axis=0)
  output2 = np.expand_dims(feature_network2.predict(np.expand_dims(image,axis=0)).flatten(),axis=0)
  output3 = np.expand_dims(feature_network3.predict(np.expand_dims(image,axis=0)).flatten(),axis=0)
  
  subSVM_output1 = np.zeros((1,10))
  subSVM_output2 = np.zeros((1,10))
  subSVM_output3 = np.zeros((1,10))
  for i in range(10):
    subSVM_output1[:,i] = subSVM1[i].predict_proba(output1)[:,1]
    subSVM_output2[:,i] = subSVM2[i].predict_proba(output2)[:,1]
    subSVM_output3[:,i] = subSVM3[i].predict_proba(output3)[:,1]
  comb_subSVM_output = np.hstack([subSVM_output1, subSVM_output2, subSVM_output3])
  final_SVM_out = np.zeros((1,10))
  for i in range(10):
#    final_SVM_out[:,i] = final_SVM[i].predict_proba(comb_subSVM_output)[:,1]
    final_SVM_out[:,i] = final_SVM[i].decision_function(comb_subSVM_output)[0]
  return final_SVM_out

def predict_DNR_tensor(image):
  image = image.numpy()[0,:,:,:]
  final_SVM_out = predict_DNR(image)
  return final_SVM_out
  
def predict_CNN(image):
  prediction = CNN_model.predict(np.expand_dims(image,axis=0))
  return prediction

image = X_train[0,:,:,:]
show_image(image)
y_pred = predict_DNR(image)

#import tensorflow.keras.losses as losses
#y_true = np.expand_dims(y_train[0,:],axis=0)
#loss = losses.squared_hinge(y_true, y_pred)
#
#from tensorflow.keras import backend as K
#grads = K.gradients(loss, CNN_model.input)[0]
#iterate = K.function([CNN_model.input], [loss, grads])
import tensorflow as tf
import tensorflow.keras.losses as losses


x = np.expand_dims(X_train[0,:,:,:],axis=0)
x = tf.convert_to_tensor(x)

with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)

  ##
  y_pred = predict_DNR_tensor(x)/2 # dividing by 2 to normalise back into [0,1 range]
  y_pred = tf.convert_to_tensor(y_pred, dtype="float32")
  ##
  
  y_pred2 = CNN_model(x)

  y_true = np.expand_dims(y_train[0,:],axis=0)
  loss = losses.squared_hinge(y_true,y_pred)
  loss2 = losses.squared_hinge(y_true,y_pred2)

gradient = tape.gradient(loss,x)
gradient2 = tape.gradient(loss2,x)
del tape
print('gradient:',gradient)
print('y_pred:',y_pred)
print('proba:',y_pred2)

#shape is different between the two
# is hingeloss not differentiable? -> not the case because it works with CNN_model(pred)