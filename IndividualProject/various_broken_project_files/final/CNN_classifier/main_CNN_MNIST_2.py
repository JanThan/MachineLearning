"""
Main CNN script
"""

# Imports ##
# general
import matplotlib.pyplot as plt
import numpy as np

# ML libraries
from tensorflow.keras.datasets import mnist

# ML utilities
from tensorflow.keras.utils import to_categorical


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
  
def load_and_encode(target_shape):
  # load dataset
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train, y_train = X_train[:10000,:,:],y_train[:10000]
  X_test, y_test =  X_test[:1000,:,:], y_test[:1000]
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

#CNN_model = train_CNN.train_model(packet, save_model = 'True')
CNN_model = load_CNN.load_model(packet)

#subSVM1, subSVM2, subSVM3, features = train_subSVMs.train(CNN_model, data, c=0.1, save_model = 'True', get_accuracies= 'True')
subSVM1, subSVM2, subSVM3, features = load_subSVMs.load(CNN_model, data, c=0.1, get_accuracies='False')

subSVMs = [subSVM1, subSVM2, subSVM3]
feature1_train, feature1_test,\
feature2_train, feature2_test,\
feature3_train, feature3_test = features

#final_SVM = train_finalSVM.train(data, features, subSVMs, number_of_classes, get_accuracies='True')
#final_SVM = load_finalSVM.load(data, features, subSVMs, number_of_classes, get_accuracies='False')
#filename = 'P:\\Implementations\\CNN_keras\\MNIST\\postGateway\\massivestoragearray'
final_SVM = joblib.load('saved_finalSVM.pkl')

"""
end of script.
Might rejig and put each line into a def so i can call  from another file
"""
adversarial_pertubations = joblib.load('P:\\Implementations\\StartingFresh\\adversarial_images.pkl')
for i in range(number_of_classes):
  adversarial_pertubations[i]=np.reshape(adversarial_pertubations[i],(28,28,1))

 
predictions = np.zeros((10,50))
for i in range(50):
  predictions[:,i] = CNN_model.predict(np.expand_dims(X_train[i,:,:,:],axis=0))

NUMBER = 48
ADVERSARY = 3
plt.imshow(np.squeeze(X_train[NUMBER,:,:,:]), cmap=plt.get_cmap('binary'))
new_adv_image = X_train[NUMBER,:,:,:] + adversarial_pertubations[ADVERSARY]
print(CNN_model.predict(np.expand_dims(new_adv_image,axis=0))  )
plt.imshow(np.squeeze(new_adv_image), cmap=plt.get_cmap('binary'), interpolation='nearest')

for i in range(10):
  dist = np.linalg.norm(adversarial_pertubations[i])
  print('dist:',dist)


# test on DNR
from load_subSVMs import get_features # #get_features(cnn_model, X_train, y_train, X_test, y_test):
features = get_features(CNN_model, np.expand_dims(new_adv_image,axis=0), y_train, X_test, y_test) 
feature1 = features[0]
feature2 = features[2]
feature3 = features[4]

subSVM1_out = np.zeros((1,10))
subSVM2_out = np.zeros((1,10))
subSVM3_out = np.zeros((1,10))
final_SVM_out = np.zeros((1,10))
for i in range(10):
  subSVM1_out[:,i] = subSVM1[i].predict_proba(feature1)[:,1]
  subSVM2_out[:,i] = subSVM2[i].predict_proba(feature2)[:,1]
  subSVM3_out[:,i] = subSVM3[i].predict_proba(feature3)[:,1]

subSVM_output = np.hstack([subSVM1_out, subSVM2_out, subSVM3_out])  
for i in range(10):
  final_SVM_out[:,i] = final_SVM[i].predict_proba(subSVM_output)[:,1]
  



"""
getting sub SVM outs below....
#"""
#features = get_features(CNN_model, X_train, y_train, X_test, y_test) 
#feature1_train = features[0]
#feature2_train = features[2]
#feature3_train = features[4]
#feature1_test = features[1]
#feature2_test = features[3]
#feature3_test = features[5]


#subSVM1_train_out = np.zeros((10000,10))
#subSVM2_train_out = np.zeros((10000,10))
#subSVM3_train_out = np.zeros((10000,10))
#subSVM1_test_out = np.zeros((1000,10))
#subSVM2_test_out = np.zeros((1000,10))
#subSVM3_test_out = np.zeros((1000,10))
#
#
#for i in range(10):
#  print('class:', i)
#  subSVM1_train_out[:,i] = subSVM1[i].predict_proba(feature1_train)[:,1]
#  subSVM2_train_out[:,i] = subSVM2[i].predict_proba(feature2_train)[:,1]
#  subSVM3_train_out[:,i] = subSVM3[i].predict_proba(feature3_train)[:,1]
#  subSVM1_test_out[:,i] = subSVM1[i].predict_proba(feature1_test)[:,1]
#  subSVM2_test_out[:,i] = subSVM2[i].predict_proba(feature2_test)[:,1]
#  subSVM3_test_out[:,i] = subSVM3[i].predict_proba(feature3_test)[:,1]
#
#massive_storage_array = [subSVM1_train_out, subSVM2_train_out, subSVM3_train_out,\
#                         subSVM1_test_out, subSVM2_test_out, subSVM3_test_out]
#filename = 'P:\\Implementations\\CNN_keras\\MNIST\\postGateway\\massivestoragearray'
#joblib.dump(massive_storage_array, filename)

"""
retraining final SVM
"""

#filename = 'P:\\Implementations\\CNN_keras\\MNIST\\postGateway\\massivestoragearray'
#subSVM1_train_out, subSVM2_train_out, subSVM3_train_out,\
#subSVM1_test_out, subSVM2_test_out, subSVM3_test_out = joblib.load(filename)
#
#from sklearn.svm import SVC
#subSVM_concat_train_out = np.hstack([subSVM1_train_out, subSVM2_train_out, subSVM3_train_out])
#finalSVM = []
#for i in range(number_of_classes):
#  finalSVM.append(SVC(C=0.1, kernel='rbf', probability=True, decision_function_shape='ovr'))
#  finalSVM[i].fit(subSVM_concat_train_out, y_train[:,i])
#  print('class:',i)
#
#joblib.dump(finalSVM, 'saved_finalSVM.pkl', compress = 3)  
#
#from load_subSVMs import get_features
##still okay
#NUMBER = 0
#features = get_features(CNN_model, np.expand_dims(X_train[NUMBER,:,:,:],axis=0), y_train, X_test, y_test) 
#feature1 = features[0]
#feature2 = features[2]
#feature3 = features[4]
#
#subSVM1_out = np.zeros((1,10))
#subSVM2_out = np.zeros((1,10))
#subSVM3_out = np.zeros((1,10))
##still okay
##final_SVM_out = np.zeros((1,10))
#for i in range(10):
#  subSVM1_out[:,i] = subSVM1[i].predict_proba(feature1)[:,1]
#  subSVM2_out[:,i] = subSVM2[i].predict_proba(feature2)[:,1]
#  subSVM3_out[:,i] = subSVM3[i].predict_proba(feature3)[:,1]
##still okay
#final_SVM_input = np.hstack([subSVM1_out, subSVM2_out, subSVM3_out])  
#for i in range(10):
#  final_SVM_out[:,i] = finalSVM[i].predict_proba(final_SVM_input)[:,1]

