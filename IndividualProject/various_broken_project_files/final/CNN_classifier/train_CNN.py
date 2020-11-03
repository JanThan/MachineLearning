# -*- coding: utf-8 -*-
"""
This script trains the core CNN model
"""
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

def def_model(train_data_shape, number_of_classes, dropout):
  # kernel regulazier-tries to reduce weights-> using L1 weight decay reduces effect of small changes --> REASON FOR LOW ACCURACIES
  # initialisation of weights is not mentioned -> using uniform distribution
  # assuming same padding
  model = Sequential()
  model.add(Conv2D(32, (3,3), activation = 'relu', padding='same', kernel_initializer='normal', input_shape=train_data_shape))
  model.add(Conv2D(32, (3,3), activation = 'relu', padding='same',kernel_initializer='normal'))
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(64, (3,3), activation = 'relu', padding='same',kernel_initializer='normal' ))
  model.add(Conv2D(64, (3,3), activation = 'relu', padding='same', kernel_initializer='normal', name='feature1'))
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(dropout))
  
  model.add(Flatten())
  model.add(Dense(200, activation = 'relu',kernel_initializer='normal', name='feature2'))
  model.add(Dense(200, activation = 'relu',kernel_initializer='normal', name='feature3'))
  
  model.add(Dense(number_of_classes, activation='softmax'))
  return model

def summarise_diagnostics(history):
  print(history.history)
  # plots and saves loss
  plt.figure(figsize=(10,10))
  plt.subplot(211)
  plt.title('Cross Entropy Loss')
  plt.xlabel('Iterations')
  plt.ylabel('Cross Entropy Loss')
  plt.plot(history.history['loss'], color='blue', label='train')
  plt.plot(history.history['val_loss'], color='orange', label='test')
  plt.legend()
  # plot accuracy
  plt.subplot(212)
  plt.title('Classification Accuracy')
  plt.xlabel('Iterations')
  plt.ylabel('Classification Accuracy')
  plt.plot(history.history['accuracy'], color='blue', label='train')
  plt.plot(history.history['val_accuracy'], color='orange', label='test')
  plt.legend()
  # save plot to file
#  filename = sys.argv[0].split('/')[-1]
#  plt.savefig(filename + 'CNN_plot.png')
  plt.savefig('CNN_plot.png')
  plt.close()


def train_model(packet, save_model):
  print('\nTraining CNN model...')
  learning_rate, momentum, dropout, batch_size, epochs, decay,\
  number_of_classes, image_shape,\
  X_train, y_train, X_test, y_test = packet
  
  
  # create model using Keras 
  model = def_model(image_shape, number_of_classes, dropout)
  opt = SGD(lr=learning_rate, decay=decay, momentum=momentum)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(X_test, y_test), verbose=1)
  # evaluate model
  _,acc = model.evaluate(X_test, y_test, verbose=0)
  print('Trained model accuracy %', acc*100)
  # learning curve
  summarise_diagnostics(history)
  if save_model == 'True':
    model.save_weights('CNN_weights.h5')
  
  return model
