# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:33:09 2020

@author: janit
"""

import os
work_dir = "P:\Implementations\StartingFresh\TensorFlow-Tutorials-master"
os.chdir(work_dir)


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import joblib

tf.reset_default_graph()

from mnist import MNIST
data = MNIST
data = MNIST(data_dir="data/MNIST/")
print("--------------------\nSize of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels

"""
Helper funtion to plot images
"""
def plot_images(images, cls_true, cls_pred=None, noise=0.0):
  assert len(images) == len(cls_true) == 9
  
  # Create figure with 3x3 sub-plots.
  fig, axes = plt.subplots(3, 3)
  fig.subplots_adjust(hspace=0.3, wspace=0.3)

  for i, ax in enumerate(axes.flat):
    # Get the i'th image and reshape the array.
    image = images[i].reshape(img_shape)
    
    # Add the adversarial noise to the image.
    image += noise
    
    # Ensure the noisy pixel-values are between 0 and 1.
    image = np.clip(image, 0.0, 1.0)

    # Plot image.
    ax.imshow(image,
              cmap='binary', interpolation='nearest')

    # Show true and predicted classes.
    if cls_pred is None:
        xlabel = "True: {0}".format(cls_true[i])
    else:
        xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

    # Show the classes as the label on the x-axis.
    ax.set_xlabel(xlabel)
    
    # Remove ticks from the plot.
    ax.set_xticks([])
    ax.set_yticks([])
  
  # Ensure the plot is shown correctly with multiple plots
  # in a single Notebook cell.
  plt.show()

"""
Get a few of these images to see if data is correct
"""
input('\n---------------\nEnter to get first 9 images from the test set...')
# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

input('\n---------------\nEnter to continue...')

"""
Placeholder Variables
"""
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1) # placeholder for class number

"""
Adversarial Noise

The pixels in the input image are float-values between 0.0 and 1.0. 
The adversarial noise is a number that is added or subtracted from the pixels 
in the input image.
"""
noise_limit = 0.30
noise_l2_weight = 0.080
ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise', trainable=False,
                      collections=collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))

x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)


"""
CNN
"""
# Start the network with the noisy input image.
net = x_noisy_image

# 1st convolutional layer.
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=32, kernel_size=3, activation=tf.nn.relu)
# 2nd convolutional layer.
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=32, kernel_size=3, activation=tf.nn.relu)
# Pooling layer
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# 3rd convolutional layer.
net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same',
                       filters=64, kernel_size=3, activation=tf.nn.relu)
# 4th convolution layer
net = tf.layers.conv2d(inputs=net, name='layer_conv4', padding='same',
                       filters=64, kernel_size=3, activation=tf.nn.relu)
# Pooling layer
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# Flatten layer.This should eventually be replaced by:
# net = tf.layers.flatten(net)
net = tf.contrib.layers.flatten(net)

# 1st fully-connected / dense layer.
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=200, activation=tf.nn.relu)
# 2nd fully-connected / dense layer.
net = tf.layers.dense(inputs=net, name='layer_fc2',
                      units=200, activation=tf.nn.relu)
# 3rd fully-connected / dense layer.
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=tf.nn.softmax)



# Unscaled output of the network.
logits = net
# Softmax output of the network.
y_pred = tf.nn.softmax(logits=logits)
# Loss measure to be optimized.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                           logits=logits)
loss = tf.reduce_mean(cross_entropy)

"""
Optimizer for Normal Training
"""
[var.name for var in tf.trainable_variables()]
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

"""
Optimizer for Adversarial Noise
"""
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
[var.name for var in adversary_variables]

l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
# combine normal loss-function with l2-loss for adversarial noise
loss_adversary = loss + l2_loss_noise

optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)

"""
Performance Measures
"""
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

""" 
----------TENSORFLOW RUN ------------------
"""

session = tf.Session()
session.run(tf.global_variables_initializer())

# initialise adversarial noise
def init_noise():
    session.run(tf.variables_initializer([x_noise]))

init_noise()   
    
"""
Helper function to perform optimization iterations
"""

train_batch_size = 64

def optimize(num_iterations, adversary_target_cls=None):
  # Start-time used for printing time-usage below.
  start_time = time.time()

  for i in range(num_iterations):
    # Get a batch of training examples.
    # x_batch now holds a batch of images and
    # y_true_batch are the true labels for those images.
    x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)

    # If we are searching for the adversarial noise, then
    # use the adversarial target-class instead.
    if adversary_target_cls is not None:
      # The class-labels are One-Hot encoded.
      
      # Set all the class-labels to zero.
      y_true_batch = np.zeros_like(y_true_batch)

      # Set the element for the adversarial target-class to 1.
      y_true_batch[:, adversary_target_cls] = 1.0
        
    # Put the batch into a dict with the proper names
    # for placeholder variables in the TensorFlow graph.
    feed_dict_train = {x: x_batch,
                       y_true: y_true_batch}

    # If doing normal optimization of the neural network.
    if adversary_target_cls is None:
      # Run the optimizer using this batch of training data.
      # TensorFlow assigns the variables in feed_dict_train
      # to the placeholder variables and then runs the optimizer.
      session.run(optimizer, feed_dict=feed_dict_train)
    else:
      # Run the adversarial optimizer instead.
      # Note that we have 'faked' the class above to be
      # the adversarial target-class instead of the true class.
      session.run(optimizer_adversary, feed_dict=feed_dict_train)
      
      # Clip / limit the adversarial noise. This executes
      # another TensorFlow operation. It cannot be executed
      # in the same session.run() as the optimizer, because
      # it may run in parallel so the execution order is not
      # guaranteed. We need the clip to run after the optimizer.
      session.run(x_noise_clip)

    # Print status every 100 iterations.
    if (i % 100 == 0) or (i == num_iterations - 1):
      # Calculate the accuracy on the training-set.
      acc = session.run(accuracy, feed_dict=feed_dict_train)

      # Message for printing.
      msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

      # Print it.
      print(msg.format(i, acc))

  # Ending time.
  end_time = time.time()

  # Difference between start and end-times.
  time_dif = end_time - start_time

  # Print the time-usage.
  print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


"""
Helper functions to get and plot noise
"""
def get_noise():
  # Run the TensorFlow session to retrieve the contents of
  # the x_noise variable inside the graph.
  noise = session.run(x_noise)

  return np.squeeze(noise)

def plot_noise():
  # Get the adversarial noise from inside the TensorFlow graph.
  noise = get_noise()
  
  # Print statistics.
  print("Noise:")
  print("- Min:", noise.min())
  print("- Max:", noise.max())
  print("- Std:", noise.std())

  # Plot the noise.
  plt.imshow(noise, interpolation='nearest', cmap='seismic',
             vmin=-1.0, vmax=1.0)
  
"""
Helper function to plot example errors
"""
def plot_example_errors(cls_pred, correct):
  # This function is called from print_test_accuracy() below.

  # cls_pred is an array of the predicted class-number for
  # all images in the test-set.

  # correct is a boolean array whether the predicted class
  # is equal to the true class for each image in the test-set.

  # Negate the boolean array.
  incorrect = (correct == False)
  
  # Get the images from the test-set that have been
  # incorrectly classified.
  images = data.x_test[incorrect]
  
  # Get the predicted classes for those images.
  cls_pred = cls_pred[incorrect]

  # Get the true classes for those images.
  cls_true = data.y_test_cls[incorrect]

  # Get the adversarial noise from inside the TensorFlow graph.
  noise = get_noise()
  
  # Plot the first 9 images.
  plot_images(images=images[0:9],
              cls_true=cls_true[0:9],
              cls_pred=cls_pred[0:9],
              noise=noise)  

"""
Helper function to plot confusion matrix
"""
def plot_confusion_matrix(cls_pred):
  # This is called from print_test_accuracy() below.

  # cls_pred is an array of the predicted class-number for
  # all images in the test-set.

  # Get the true classifications for the test-set.
  cls_true = data.y_test_cls
  
  # Get the confusion matrix using sklearn.
  cm = confusion_matrix(y_true=cls_true,
                        y_pred=cls_pred)

  # Print the confusion matrix as text.
  print(cm)

"""
Helper function to show the performance
"""
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

  # Number of images in the test-set.
  num_test = data.num_test

  # Allocate an array for the predicted classes which
  # will be calculated in batches and filled into this array.
  cls_pred = np.zeros(shape=num_test, dtype=np.int)

  # Now calculate the predicted classes for the batches.
  # We will just iterate through all the batches.
  # There might be a more clever and Pythonic way of doing this.

  # The starting index for the next batch is denoted i.
  i = 0

  while i < num_test:
      # The ending index for the next batch is denoted j.
      j = min(i + test_batch_size, num_test)

      # Get the images from the test-set between index i and j.
      images = data.x_test[i:j, :]

      # Get the associated labels.
      labels = data.y_test[i:j, :]

      # Create a feed-dict with these images and labels.
      feed_dict = {x: images,
                   y_true: labels}

      # Calculate the predicted class using TensorFlow.
      cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

      # Set the start-index for the next batch to the
      # end-index of the current batch.
      i = j

  # Convenience variable for the true class-numbers of the test-set.
  cls_true = data.y_test_cls

  # Create a boolean array whether each image is correctly classified.
  correct = (cls_true == cls_pred)

  # Calculate the number of correctly classified images.
  # When summing a boolean array, False means 0 and True means 1.
  correct_sum = correct.sum()

  # Classification accuracy is the number of correctly classified
  # images divided by the total number of images in the test-set.
  acc = float(correct_sum) / num_test

  # Print the accuracy.
  msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
  print(msg.format(acc, correct_sum, num_test))

  # Plot some examples of mis-classifications, if desired.
  if show_example_errors:
      print("Example errors:")
      plot_example_errors(cls_pred=cls_pred, correct=correct)

  # Plot the confusion matrix, if desired.
  if show_confusion_matrix:
      print("Confusion Matrix:")
      plot_confusion_matrix(cls_pred=cls_pred)

"""
Normal Optimization of neural network
"""
# training our neural network model here
input('\n---------------\nEnter to train the neural network...')
optimize(num_iterations=1000)

print_test_accuracy(show_example_errors=True)


"""
Find the adversarial noise
"""
# optimizing stage uses Adam Solver to sort out the gradient descent stuff for us. 
# I have wrapped it all up in the one optimize function
input('\n---------------\nEnter to find the adversarial noise for target class 3...')
init_noise()

optimize(num_iterations=1000, adversary_target_cls=3)
plot_noise()
print('\nRed -> positive noise values, Blue -> negative noise values')
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

"""
Adversarial noise for all target classes
"""
input('\n---------------\nEnter to find adversarial noise for all target classes...')
def find_all_noise(num_iterations=1000):
  # Adversarial noise for all target-classes.
  all_noise = []

  # For each target-class.
  for i in range(num_classes):
      print("Finding adversarial noise for target-class:", i)

      # Reset the adversarial noise to zero.
      init_noise()

      # Optimize the adversarial noise.
      optimize(num_iterations=num_iterations,
               adversary_target_cls=i)

      # Get the adversarial noise from inside the TensorFlow graph.
      noise = get_noise()

      # Append the noise to the array.
      all_noise.append(noise)

      # Print newline.
      print()
  
  return all_noise

all_noise = find_all_noise(num_iterations=300)
file_name = 'P:\\Implementations\\StartingFresh\\adversarial_images.pkl'
joblib.dump(all_noise, file_name, compress=3)

"""
Plot the adversarial noise for all target classes
"""
def plot_all_noise(all_noise):    
  # Create figure with 10 sub-plots.
  fig, axes = plt.subplots(2, 5)
  fig.subplots_adjust(hspace=0.2, wspace=0.1)

  # For each sub-plot.
  for i, ax in enumerate(axes.flat):
      # Get the adversarial noise for the i'th target-class.
      noise = all_noise[i]
      
      # Plot the noise.
      ax.imshow(noise,
                cmap='seismic', interpolation='nearest',
                vmin=-1.0, vmax=1.0)

      # Show the classes as the label on the x-axis.
      ax.set_xlabel(i)

      # Remove ticks from the plot.
      ax.set_xticks([])
      ax.set_yticks([])
  
  # Ensure the plot is shown correctly with multiple plots
  # in a single Notebook cell.
  plt.show()

plot_all_noise(all_noise)

input('\n---------------\nEnter to begin training CNN to withstand the above adversarial noise...')
"""
Now train the CNN to be immune to the noise found to missclassify to each number
"""
# we can  do this by retraining the neural network to be less sensitive to the adversarial patterns found

"""
Helper function to make neural network immune to noise
"""
def make_immune(target_cls, num_iterations_adversary=700,
                num_iterations_immune=500):

  print("Target-class:", target_cls)
  print("Finding adversarial noise ...")

  # Find the adversarial noise.
  optimize(num_iterations=num_iterations_adversary,
           adversary_target_cls=target_cls)

  # Newline.
  print()
  
  # Print classification accuracy.
  print_test_accuracy(show_example_errors=False,
                      show_confusion_matrix=False)

  # Newline.
  print()

  print("Making the neural network immune to the noise ...")

  # Try and make the neural network immune to this noise.
  # Note that the adversarial noise has not been reset to zero
  # so the x_noise variable still holds the noise.
  # So we are training the neural network to ignore the noise.
  optimize(num_iterations=num_iterations_immune)

  # Newline.
  print()
  
  # Print classification accuracy.
  print_test_accuracy(show_example_errors=False,
                      show_confusion_matrix=False)


"""
Make immune to noise for target class 3
"""
input('\n---------------\nFirst making immune to target class 3...Enter to continue...')
print('Training to be immune')
make_immune(target_cls=3)

# run it again and now it is much harder to find adversarial noise for target class 3
print('Seeing again... should be much harder now...')
make_immune(target_cls=3)


"""
Make immune to all target classes (double runs)
"""
input('\n---------------\nEnter to make immune to all the target classes in the same way as previous...')
for i in range(10):
  
  make_immune(target_cls=i)
  # Print newline.
  print()
  make_immune(target_cls=i)
  # Print newline.
  print('\n---------------\n')
# this doesnt work too well. It seems that making a model resistant to one type of misclassification, 
# resets the missclassification of other classes

"""
Plot the adversarial noise
"""
input('\n---------------\nEnter to plot the optimised noise')  
plot_noise()
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
# neural network should have maybe a slightly higher accuracy on dirty images 

""" 
What is the performance on clean images
"""
input('\n---------------\nEnter for performance on clean images...')
# initialise some noise
init_noise()
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
# neural network might perform worse on clean images now though after being retrained on dirty images

"""
Close tensorflow session
"""
session.close()












