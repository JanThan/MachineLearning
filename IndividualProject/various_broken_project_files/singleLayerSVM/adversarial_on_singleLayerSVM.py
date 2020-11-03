# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:00:50 2020

@author: janit
"""
# Imports ##
# general
import matplotlib.pyplot as plt
import numpy as np
import joblib
from training_singleLayerSVM import flatten
from load_singleLayerSVM import load_model

# ML utilities
from tensorflow.keras.losses import categorical_crossentropy # another name for multiclass log loss
from sklearn.metrics import hinge_loss

# Python scripts used

def show_data_example(i, dataset, label):
  # show some of the images in the dataset
  # call multiple times for multiple images
  # squeeze is necessary here to get rid of the extra dimension introduced in rehsaping
  print('\nExample Image: %s from selected dataset' %i)
  plt.imshow(np.squeeze(dataset[i]), cmap=plt.get_cmap('gray'))
  plt.show()
  print('label for image:', label[i,:])
  return  
  


  
print('loading presaved packets and feature data')
packet = joblib.load('P:\Implementations\CNN_keras\MNIST\postGateway\data_packet.pkl')
  
learning_rate, momentum, dropout, batch_size, epochs, decay,\
number_of_classes, image_shape, \
X_train, y_train, X_test, y_test = packet

# flatten data for input to SVM
X_train_flat = flatten(X_train)
X_test_flat = flatten(X_test)

"""
Training Adversarial Image
"""

epochs = 1 # epochs for training adversary
prediction = np.zeros((1,number_of_classes))
prediction_proba = np.zeros((number_of_classes,2))
predicted_decision = np.zeros((1,number_of_classes))
loss = 0

# show the image before turning to adversary
show_data_example(5, X_train, y_train)
# get SVM model
model = load_model()
for i in range(number_of_classes):
  prediction[0,i]=model[i].predict([X_train_flat[5,:]])
  prediction_proba[i,:]=model[i].predict_proba([X_train_flat[5,:]])
  predicted_decision[0,i] = model[i].decision_function([X_train_flat[5,:]])
  loss = loss + hinge_loss([y_train[5,i]],predicted_decision[:,i])
  
  
print('prediction:',prediction)  
print('prediction probabilities:\n',prediction_proba)
print('decision function:', predicted_decision) # decision function sgn(sum(y*alpha*K(x,x') + p))
print('loss:',loss)

"""
actual training of adversarial
turn this into definition when more convenient
"""





# get values of decision function
num_sup_vecs = model[0].n_support_
sup_vecs = model[0].support_vectors_
dual_coef = model[0].dual_coef_
intercept = model[0].intercept_

print('.....ADVERSARIAL STUFF NOW.....')





# find the maximum score - CORRECT SCORE - min this score to maximise adversary
index_corr = np.where(prediction_proba == np.amax(prediction_proba[:,1]))
print('\nCorrect_Score',index_corr, np.amax(prediction_proba[:,1]))

# find the next highest score - INCORRECT SCORE- max this score (max the adversary)
index_adv = np.where(prediction_proba == np.partition(prediction_proba[:,1].flatten(), -2)[-2])
print('Highest_NOT_Score',index_adv, np.partition(prediction_proba[:,1].flatten(), -2)[-2],'\n')

# redefining indexes to put straight into SVM model for the specific decision score
index_corr = index_corr[0][0] 
index_adv = index_adv[0][0]

image = np.expand_dims(X_train_flat[5,:],axis=0)
label = np.expand_dims(y_train[5,:], axis=0)

def gradient_wrt_input(image, image_,adv):
  if adv==True:
    decision_function = model[index_adv].decision_function(image)
    decision_function_= model[index_adv].decision_function(image_)
  else:
    decision_function = model[index_corr].decision_function(image)
    decision_function_= model[index_corr].decision_function(image_)
  
  grad =  decision_function_ - decision_function # moves in direction of increasing loss
  
#  print('image_:', model[index_adv].decision_function(image_))
#  print('image:', model[index_adv].decision_function(image))
#  
  hing = hinge_loss(label[:,index_adv],decision_function)
  hing_= hinge_loss(label[:,index_adv],decision_function_)
#  print('hing:',hing,'\nhing_:',hing_)
  return np.sign(grad), hing, hing_


epochs = 1
learning_rate = 0.001
error = 8
x = image
x_ = x * 255
x_=x+np.random.rand(1,784)
x_ = x_/255

for j in range(784):
  
  for i in range(epochs):
    pertubation, hing, hing_ = gradient_wrt_input(x, x_,False)
#    print('pertubation:', pertubation)
    x_[:,j] = x_[:,j] + pertubation*learning_rate
    
    dist = np.linalg.norm((x-x_), ord=2)
#    print('dist:',dist)
  #  # calculate euclidian distance
  #  dist = np.linalg.norm((x-x_), ord=2)
  #  print("distance between adversary and original, ||x-x'||:", dist) 
  #  noise = (error*(x-x_))/(np.maximum(error,dist))
  #  print('noise:',np.amax(noise))
    
    x_ = x + (error*(x-x_))/(np.maximum(error,dist))
#  print('dist:',dist,'pertubation:',pertubation)
#  print('hing:',hing,'hing_:',hing_)
  
show_data_example(0,x_.reshape((1,28,28,1)),label)

#if __name__ == "__main__":
#    main()  
#    
  
for i in range(number_of_classes):
  prediction[0,i]=model[i].predict(x_)
  prediction_proba[i,:]=model[i].predict_proba(x_)
  predicted_decision[0,i] = model[i].decision_function(x_)

print(prediction)
print(prediction_proba)
print(predicted_decision)

# find the maximum score - CORRECT SCORE - min this score to maximise adversary
index_corr = np.where(prediction_proba == np.amax(prediction_proba[:,1]))
print('\nCorrect_Score',index_corr, np.amax(prediction_proba[:,1]))

# find the next highest score - INCORRECT SCORE- max this score (max the adversary)
index_adv = np.where(prediction_proba == np.partition(prediction_proba[:,1].flatten(), -2)[-2])
print('Highest_NOT_Score',index_adv, np.partition(prediction_proba[:,1].flatten(), -2)[-2],'\n')