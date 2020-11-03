
"""
simple neural network to find if people are diabetic or not tutorial...
one input layer and one output layer

DATA:
Person    Smoking   Obesity   Exercise  Diabetic
1         0           1           0          1
2         0           0           1          0
3         1           0           0          0
4         1           1           0          1
5         1           1           1          1

==> 3 nodes in input layer
==> 1 node in output layer

It is obvious that obesity in this dataset strongly correlates with diabetic 
Task is to create a NN capable of prediccting this about an unknown person

A neural network executes in two step: 'feed-forward' and 'back' propagation
FEED-FORWARD ------------------------------------------------------------------
predictions made are based on values of input nodes and weights
  Step 1:
    - Calculate dot product between inputs and weights
    - X.W, where X0 = 1 and W0 = bias term
  Step 2:
    - Pass result from Step 1 through to an activation function
    - Example of activation function is the sigmoid function:
      sigmoid = theta(X.W) = 1/(1+exp(-X.W))

BACK --------------------------------------------------------------------------
In the beginning before any training, neural network makes random predictions:
  - lots of error
We start by letting NN make random  predictions, then we compare predicted with 
actual output.
Next we fine tune the weights in a way that predicted outputs become closer to 
actual output.
AKA Training the NN 
Back propagation is in effect the training of the NN. 
  Step 1:
    - Calculate cost
    - Several ways to compute cost, here mean squared error cost function used
    - MSE = 1/n * SUM(predicted-observed)^2 for all n 
        where n is the number of observations
  Step 2:
    - Minimising the cost...smaller the cost, the better the more correct the 
    predictions and consequently the better the model
    - Effectively an optimisation problem to find the function minima
    - Use gradient descent to find this minima
    - repeat until convergence:
        wj = wj - alpha * dJ(w0, w1 ... wn)
        --> find partial derivative of cost function with respect to weights
        --> subtract this from existing weight to get new weight
        --> Note: alpha is the learning rate, wj is the jth weight
        --> repeat this until weights stop changing...ie convergence        
"""

import numpy as np

# our activation function
def sigmoid(x):
  return 1/(1+np.exp(-x))
# derivative of activation function
def sigmoid_der(x):
  return sigmoid(x)*(1-sigmoid(x))

# recreating data set described above
feature_set = np.array(([0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]))
labels = np.array([1,0,0,1,1])
labels = labels.reshape(5,1)
(m,n) = feature_set.shape

X = np.hstack((np.ones((m,1)),feature_set))
y = labels

# define hyper-parameters for neural network
np.random.seed(42) # setting seed to 7 to ensure the same values for comparison
theta = np.random.rand(n+1,1) # n x 1
alpha = 0.05

# MAIN TRAINING
iterations = 20000
N = len(X)
for epoch in range(iterations):
  # feed-forward step 1

  Xtheta = np.dot(X, theta)
  # feed-forward step 2
  z = sigmoid(Xtheta) # mx1
  
  # back propagation step 1
  error = z - y                   # (predicted - observed)
  J_cost = 1/N * sum(error**2)
  print('Iteration:',epoch)
  print('Total error:', np.around(error.sum(),4))
  print('--------------------')
  # back propagation step 2
  # dcost_dw = dcost_dpred * dpred_dz * dz_dw
  # *dcost_dpred --> dJ = 2/N * (predicted-observed)
  # *dpred_dz --> sigmoid_der(z) (derivative of sigmoid function)
  # *dz_dw --> z = XW+b --> dz_dw = X
  dcost_dpred = 2/N * error
  dpred_dz = sigmoid_der(z)
  dz_dw = X
  dcost_dw = np.dot(dz_dw.T, (dcost_dpred * dpred_dz))
  
  theta -= alpha * dcost_dw


# Testing
test_point = np.array([1,0,0])
print('Test Point:',test_point)
test_point = np.append(1,test_point)
result = np.around(sigmoid(np.dot(test_point, theta)),4)
print('Confidence in being Diabetic (%):',result*100)

test_point = np.array([0,1,0])
print('Test Point:',test_point)
test_point = np.append(1,test_point)
result = np.around(sigmoid(np.dot(test_point, theta)),4)
print('Confidence in being Diabetic (%):',result*100)