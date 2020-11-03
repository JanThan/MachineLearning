# -*- coding: utf-8 -*-
"""
Loads pre trained CNN
"""
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

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


def load_model(packet):
  print('\nLoading CNN model...')
  learning_rate, momentum, dropout, batch_size, epochs, decay,\
  number_of_classes, image_shape,\
  X_train, y_train, X_test, y_test = packet
  
  opt = SGD(lr=learning_rate, decay=1e-6, momentum = momentum)
  model = def_model(image_shape, number_of_classes, dropout)
  
  model.load_weights('CNN_weights.h5')
  
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  _,acc = model.evaluate(X_test, y_test, verbose=0)
  print('Loaded model accuracy %:', acc*100)
  return model
