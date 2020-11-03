"""
Expanding on the previously built simple neural network by adding extra 
hidden layers so as to tackle more complex tasks such as:
  - image classification 
  - stock market analysis
The previous model is not capable of tackling non-linear challenges, 
by including extra hidden layers, I hope to be able to find more non-linear 
boundaries. 

I will use the Scikit Learn library to generate a non-linear dataset: 
  - specifically making use of the make_moons function to create two half-circles
  which CANNOT be matched to a linear boundary.

Regardless of the number of extra hidden layers are added, 
the basic working principle remains the same.
  
Feed forward ------------------------------------------------------------------
  
  X = [x11 x12 ... x1n    W = [w11 w12 ... w1d    
       x21 x22 ... x2n         w21 w22 ... w2d
       .    .   .   .           .    .   .   .
       .    .   .   .           .    .   .   .
       .    .   .   .           .    .   .   .
       xs1 xs2 ... xsn]        wn1 wn2 ... wnd]
   
  Z = [z11 z12 ... z1d
       z21 z22 ... z2d
        .    .   .   .   
        .    .   .   .   
        .    .   .   .   
       zs1 zs2 ... zsd]

  s -> samples
  n -> features
  d -> hidden layer nodes
  X -> matrix of all observations
  W -> weight matrix between all nodes
  Zsxd = Xsxn . Wnxd 
  
  Z1 = X x W1
  A1 = sigmoid(Z1) = sigmoid(XW1+b)
  Note: we will have an extra column of 1s in X matrix. 
  --> 1st column of W matrix will represent the added bias terms
  Z2 = A1 x W2
  A2 = sigmoid(Z2) to find at output layer
  
Backward ----------------------------------------------------------------------
  Update the weights as such:
    W1 = W1 - alpha*delta1
    W2 = W2 - alpha*delta2
    
    delta1: the cost function gradient, dJ_dw
    delta2: the difference between the immediate output and the observed
    
    delta1:     
      dJ_dw = dJ_dA2 * dA2_dZ2 * dZ2_dW2
        --> dJ_dA2 = delta2 = (predicted - observed) = A2 - y 
        Recall, A2 = sigmoid(Z2)
        --> dA2_dZ2 = sigmoid_der(Z2) = sigmoid(Z2) * (1-sigmoid(Z2))
        Recall, Z2 = A1 x W2
        --> dZ2_dW2 = A1
    delta2:
      delta2 = A2 - y
      
  Then updating the weights is simply plugging this in
  W2 -= alpha * A1' x delta2
  W1 -= alpha  * X' x delta1
  
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# generating dataset
np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.1)
(m,n) = feature_set.shape

plt.figure(figsize=(3,3))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.Spectral)
plt.show()
input('Press enter to begin training...')
labels = labels.reshape(m,1)

# parameters
samples = m
features = n
hidden_nodes = 5  # N, number of nodes in hidden layers
alpha = 0.01   # learning rate
classes = 1       # c, classes can be 1 here as we only have 1 or 0 in y
iterations = 1000  # number of iterations
costs = []        # array initialised for cross entropy loss graph

# adding bias terms
X = np.hstack((np.ones((m,1)),feature_set))   # m x (n+1)
y = labels                                    # m x c
W1 = np.random.rand(features+1, hidden_nodes) # (n+1) x N
W2 = np.random.rand(hidden_nodes, classes)    # N x c

# activation function
def sigmoid(x):
  return 1/(1+np.exp(-x))
# derivative of activation function
def sigmoid_der(x):
  return sigmoid(x)*(1-sigmoid(x))

for epoch in range(iterations):
  # forward pass
  Z1 = np.dot(X,W1)
  A1 = sigmoid(Z1)
  Z2 = np.dot(A1,W2)
  A2 = sigmoid(Z2)    # predicted
  
  # backward pass
  delta2 = A2 - y
  delta1 = np.dot(delta2, W2.T) * sigmoid_der(Z1)
  
  W2 -= alpha * np.dot(A1.T,delta2)
  W1 -= alpha * np.dot(X.T, delta1)
  
  # calculating loss
  sum_score = sum(y * np.log10(1e-15 + A2)) # small value added to avoid getting 0
  loss = 1/m * sum_score
  loss = np.sqrt(loss**2)  # forced positive
  costs.append(loss)
  print('Iteration:', epoch,'Loss:', loss) 
  
  if epoch == iterations-1:
    print('Training complete...')
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
