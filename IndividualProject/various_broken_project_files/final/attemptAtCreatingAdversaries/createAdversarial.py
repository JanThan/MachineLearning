# Imports ##
# general
import matplotlib.pyplot as plt
import numpy as np

# ML libraries
from tensorflow.keras.datasets import mnist

# ML utilities
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy # another name for multiclass log loss
from tensorflow.keras import backend as K

# Python scripts used
import train_CNN
import load_CNN
import train_subSVMs
import load_subSVMs
import train_finalSVM
import load_finalSVM
#import main_CNN_MNIST

import joblib

def show_data_example(i, dataset, label):
  # show some of the images in the dataset
  # call multiple times for multiple images
  # squeeze is necessary here to get rid of the extra dimension introduced in rehsaping
  print('\nExample Image: %s from selected dataset' %i)
  plt.imshow(np.squeeze(dataset[i]), cmap=plt.get_cmap('gray'))
  plt.show()
  print('label for image:', label[i,:])
  return

packet = joblib.load('data_packet.pkl')
features = joblib.load("features.pkl")

feature1_train = features[0]
feature2_train = features[2]
feature3_train = features [4]
feature1_test = features[1]
feature2_test = features[3]
feature3_test = features [5]

learning_rate, momentum, dropout, batch_size, epochs, decay,\
number_of_classes, image_shape, \
X_train, y_train, X_test, y_test = packet
data = [X_train, y_train, X_test, y_test]

CNN_model = load_CNN.load_model(packet)
subSVM1, subSVM2, subSVM3, features = load_subSVMs.load(CNN_model, data, c=0.1, get_accuracies='False')

subSVMs = [subSVM1, subSVM2, subSVM3]
feature1_train, feature1_test,\
feature2_train, feature2_test,\
feature3_train, feature3_test = features

final_SVM = load_finalSVM.load(data, features, subSVMs, number_of_classes, get_accuracies='False')

"""
picking and displaying a random sample from the dataset. 
storing this as original sample, x0 and label y0
labels are as such: 0 1 2 3 4 5 6 7 8 9 
r will also be included as an if statement, where r is rejection class for label scores to exceed to provide a reading

"""
print('X_train, X_test:', X_train.shape, X_test.shape)
print('y_train, y_test:', y_train.shape, y_test.shape)
show_data_example(5, X_train, y_train)

# expanding dims for compatibility with what CNN model expects
x0 = np.expand_dims(X_train[5,:,:], axis=0)
y0 = np.expand_dims(y_train[5,:], axis = 0) 
print('x0, y0 shape:', x0.shape, y0.shape)

"""
run a prediction first and get its predicted label to compare to its true label
"""

y_pred = CNN_model.predict(x0) 


######################################
######################################
######################################


import tensorflow as tf
def adversarial_pattern(image, label):
  image = tf.cast(image, tf.float32)
  with tf.GradientTape() as tape:
    tape.watch(image)
    prediction = CNN_model(image)
    loss = tf.keras.losses.hinge(label, prediction)
  gradient = tape.gradient(loss, image)
  signed_grad = tf.sign(gradient)
  return signed_grad

import random
img_rows = 28
img_cols = 28
channels = 1
x_train = X_train

def generate_adversarials(batch_size):
    x = []
    y = []
    for batch in range(batch_size):
        if batch_size > 10000 and batch % 1000 == 0:
            print(batch/batch_size)
        N = random.randint(0, 100)

        label = y_train[N]
        
        
        perturbations = adversarial_pattern(x_train[N].reshape((1, img_rows, img_cols, channels)), label).numpy()
        print(np.amax(perturbations))
        image = x_train[N]
        
        learning_rate = 0.1
        adversarial = image + perturbations * learning_rate
        
        x.append(adversarial)
        y.append(y_train[N])
    
    
    x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
    y = np.asarray(y)
    
    return x, y, perturbations

 
x_adversarial_train, y_adversarial_train, p = generate_adversarials(1)
show_data_example(0,x_adversarial_train,y0)
#x_adversarial_test, y_adversarial_test = next(generate_adversarials(10000))  
  

# Assess base model on adversarial data vs before adversarial
#print("Base accuracy on adversarial images:", CNN_model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
#print("Base accuracy on normal images:", CNN_model.evaluate(x=X_train, y=y_train, verbose=0))

#predict = CNN_model.predict(np.expand_dims(x_adversarial_train[5,:,:,:],axis=0))
#print(predict)



"""
"""
## Learn from adversarial data
#model.fit(x_adversarial_train, y_adversarial_train,
#          batch_size=32,
#          epochs=10,
#          validation_data=(x_test, y_test))
#
## Assess defended model on adversarial data
#print("Defended accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
#
## Assess defended model on regular data
#print("Defended accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
