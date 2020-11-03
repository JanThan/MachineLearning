import sys
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

"""
3 VGG-based architecture
+
Dropout Regularisation - avoid overfitting
+
Weight decay - penalise model proportional to the size of the model weights
... done through 'kernel_regularizer'
+
Batch Normalisation after each layer - effort to stabalize learning & 
accelerate learning
"""

# load dataset and encode data
def load_and_encode():
  (X_train,y_train), (X_test, y_test) = cifar10.load_data()
  # summarize loaded dataset
  print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
  print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
  # one hot encoding
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  return X_train, y_train, X_test, y_test

# because of pixel values in range [0,255] , scaling to value range [0, 1]
def normalise_pixels(train, test):
  normalised_train = train/255.0
  normalised_test = test/255.0
  return normalised_train, normalised_test


"""
[Conv *2 -> Pooling -> Drop Out (20%)] * 3 -> Flatten -> FC

"""
def def_model(X_train_shape, number_of_classes):
  model = Sequential()
  
  model.add(Conv2D(32, (3,3), activation = 'relu',kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001), input_shape=X_train_shape[1:]))
  model.add(BatchNormalization())
  model.add(Conv2D(32, (3,3), activation = 'relu',kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.2))
  
  model.add(Conv2D(64, (3,3), activation = 'relu',kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3,3), activation = 'relu',kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.3))
  
  model.add(Conv2D(128, (3,3), activation = 'relu',kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3,3), activation = 'relu',kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.4))
  
  model.add(Flatten())
  model.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001)))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))  
  model.add(Dense(number_of_classes, activation='softmax'))
  
  # compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model
##  
### plot diagnostic learning curves
def summarize_diagnostics(history):
  # plot loss
#  history_dict = history.history
#  print(history_dict.keys())
  plt.figure(figsize=(10,10))
  plt.subplot(211)
  plt.title('Cross Entropy Loss')

  plt.plot(history.history['loss'], color='blue', label='train')
  plt.plot(history.history['val_loss'], color='orange', label='test')
  # plot accuracy
  plt.subplot(212)
  plt.title('Classification Accuracy')
  plt.plot(history.history['acc'], color='blue', label='train')
  plt.plot(history.history['val_acc'], color='orange', label='test')
  # save plot to file
  filename = sys.argv[0].split('/')[-1]
  plt.savefig(filename + '_plot.png')
  plt.close()
    
def run_test():
  # load and encode dataset
  X_train, y_train, X_test, y_test = load_and_encode()
  # normalise
  X_train, X_test = normalise_pixels(X_train, X_test)
  X_train_shape = X_train.shape
  # create model using keras
  model = def_model(X_train_shape,number_of_classes=10)  
  # fit model
  history = model.fit(X_train, y_train, epochs=100, batch_size=64, 
                      validation_data=(X_test, y_test), verbose=1)
  # evaluate model
  _,acc = model.evaluate(X_test, y_test, verbose=0)
  print('accuracy percentage:', acc*100)
  # learning curve
  summarize_diagnostics(history)
  
##############################################################################  
def save_model():
  # load and encode dataset
  X_train, y_train, X_test, y_test = load_and_encode()
  # normalise
  X_train, X_test = normalise_pixels(X_train, X_test)
  X_train_shape = X_train.shape
  # create model using keras
  model = def_model(X_train_shape,number_of_classes=10)  
  # fit model
  model.fit(X_train, y_train, epochs=100, batch_size=64, 
            validation_data=(X_test, y_test), verbose=1)
  model.save('model_cifar10.h5')
##############################################################################  
def use_saved_model():
  # load and encode dataset
  X_train, y_train, X_test, y_test = load_and_encode()
  # create model using keras
  model = load_model('model_cifar10.h5')
  # evaluate model on test dataset
  _,acc = model.evaluate(X_test, y_test, verbose=0)
  print('accuracy percentage:', acc*100)

##############################################################################
def load_image(file):
  # load image
  img = load_img(file, target_size=(32,32))
  # convert to array
  img = img_to_array(img)
  # reshape
  img = img.reshape(1,32,32,3)
  # prepare pixels
  img = img.astype('float32')
  img = img/255
  return img

def predict():
  # load test image
  img_dict = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
  model = load_model('model_cifar10.h5')
  for i in range(10):
    print('loaded image:',img_dict[i])
    img = load_image('img_'+img_dict[i]+'.png')
    result = model.predict_classes(img)
    print('model predicts:',img_dict[result[0]]+'\n')

##############################################################################  

#run_test()         # run and evalute model.
#save_model()       # run and save the current model without evaluation
#use_saved_model()  # load and evaluate a saved model
predict()           # use a saved model to predict for a loaded image
