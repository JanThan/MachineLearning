import numpy as np
import matplotlib.pyplot as plt

# generate random set of data
np.random.seed(0)
X = np.random.rand(200,1)
y = 5 + 2*X+np.random.rand(200,1)

# plot
plt.scatter(X,y,s=5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# TRAINING LINEAR REGRESSION with Gradient Descent

# Random Initilalisation
# Partial Derivatives
# Update Parameters

m = len(y) # number of training examples

# Cost and Gradient Descent
X0 = np.ones((m,1))
X = np.hstack((X0,X)) # adding a column of ones to X
theta = np.zeros((2,1)) # initialise fitting parameters

# Gradient Descent settings
iterations = 1500
alpha = 0.01
print("Iterations set to:", iterations,
      "\nAlpha set to:", alpha)
# Compute Cost
print("Testing Cost Function")
h = X.dot(theta) # mx2 * 2*1 = mx1 # hypothesis
J = 1/(2*m) * np.sum((h-y)**2) # cost function
print("Cost computed is:", J)
# Gradient Descent
J_hist = np.zeros((iterations,1))   #
for i in range(iterations):
  h = X.dot(theta)
  SUM = X.transpose().dot((h-y))
  theta = theta - alpha/m * SUM
  J_hist[i] = 1/(2*m) * np.sum((h-y)**2)
print("Theta is:\n",theta)

plt.plot(J_hist)
plt.xlabel('iterations')
plt.ylabel('Cost, J')
plt.show()


plt.scatter(X[:,1],y,s=5)
plt.plot(X[:,1],X.dot(theta))
plt.xlabel('x')
plt.ylabel('y')
plt.show()


