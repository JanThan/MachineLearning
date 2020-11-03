import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): 
  return 1/(1+np.exp(-z))

def sigmoid_der(z): 
  return sigmoid(z)*(1-sigmoid(z))

def calculate_cost(y,outs): # calculating cross entropy loss
  C = -(y*np.log10(outs["A2"]**2)+(1-y)*np.log10(1-outs["A2"]**2))
  return C

# data
X = np.array(([0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]))
y = np.array([1,0,0,1,1])
y = y.reshape(5,1)

# dimensions of data
(m,nx) = X.shape
ny = y.shape[1]

# NN configuration
Nh = 5       # hidden layer nodes
iterations = 20000
alpha = 0.05      # learning rate
cost_graph = []   # for visualising loss against iterations
np.random.seed(7) # setting random see for comparison purposes

# initialising weights and bias'
params = { "W1": np.random.rand(nx, Nh),
           "b1": np.zeros((1, Nh)),
           "W2": np.random.rand(Nh, ny),
           "b2": np.zeros((1, ny))
          }

def feed_forward(X,params):
  outs = {}
  
  outs["A0"] = X
  outs["Z1"] = np.dot(outs["A0"],params["W1"])+params["b1"]
  outs["A1"] = sigmoid(outs["Z1"])
  outs["Z2"] = np.dot(outs["A1"],params["W2"])+params["b2"]
  outs["A2"] = sigmoid(outs["Z2"])
  return outs

def calculate_derivatives(params, outs, y):
  der = {}
  error_2 = (outs["A2"]-y)*sigmoid_der(outs["Z2"]) # output layer
  error_1 = np.dot(error_2, params["W2"].T) *  sigmoid_der(outs["Z1"])   
  der["dW2"] = np.dot(error_2.T, outs["A1"]).T
  der["dW1"] = np.dot(error_1.T, outs["A0"]).T
  der["db2"] = np.sum(error_2, 0)
  der["db1"] = np.sum(error_1, 0)
  return der

for epoch in range(iterations):
  # forward pass
  outs = feed_forward(X,params)
  # calculate cross entropy loss
  C = calculate_cost(y,outs)
  # backward pass
  derivatives = calculate_derivatives(params, outs, y)
  # update weights
  params["W2"] -= alpha * derivatives["dW2"]
  params["b2"] -= alpha * derivatives["db2"]
  params["W1"] -= alpha * derivatives["dW1"]
  params["b1"] -= alpha * derivatives["db1"]

  # cost visualisations
  print('iteration:',epoch,'cost:', np.around(C.sum(),4))
  cost_graph.append(C.sum())
  
plt.plot(cost_graph)
print('calculated loss:', 1/m * sum((outs["A2"]-y)**2))
