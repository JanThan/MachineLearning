# Boston Housing Prices Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


loaded_dataset = load_boston()
print(loaded_dataset.keys()) # check what's inside

#print(loaded_dataset.DESCR) # just to check any extra info about dataset

fullDataset = pd.DataFrame(loaded_dataset.data, columns = loaded_dataset.feature_names)
fullDataset['MEDV'] = loaded_dataset.target # add target values for later comparison

print(fullDataset.head(5)) # check what data looks like

print(fullDataset.isnull().sum()) # to check if there is any missing data

# applying a heatmap to quickly check correlations between data.
# most interested with strong correlations to output MEDV 
correlation_matrix = fullDataset.corr().round(1)
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(data=correlation_matrix,annot=True, ax=ax, cmap='coolwarm') 
plt.show()
# LSTAT and RM seem to have strongest correlations from heatmap
# These can be our chosen features then...
features = ['RM','LSTAT']
y = fullDataset['MEDV']
m = len(y)
X = np.zeros((m,2))
X[:,0] = fullDataset['RM']
X[:,1] = fullDataset['LSTAT']

plt.scatter(X[:,0],y,s=5)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()
plt.scatter(X[:,1],y,s=5)
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

# MODELING
# as there are multiple features, should first normalise to avoid bias
mu = np.mean(X,0)
sigma = np.std(X,0)
X_norm = (X-mu)/sigma

# redefine m for X_train
m = len(X)
X0 = np.ones((m,1))
X = np.hstack((X0,X_norm)) # adding a column of ones to Xtrain
theta = np.zeros((3,1)) # initialise fitting parameters, 2 features and constant 
y = (np.array([y])).transpose()

# split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=7)



# alpha and iterations 
alpha = 0.001
iterations = 15000
print("Chosen alpha:",alpha,
      "\nChosen iterations:",iterations)
# Gradient Descent
J_hist = np.zeros((iterations,1))
for i in range(iterations):
  h = X_train.dot(theta) # 404x3 * 3x1 = 404x1
  SUM = X_train.transpose().dot(h-y_train)
  theta = theta - alpha/m * SUM
  J_hist[i] = 1/(2*m) * np.sum((h-y_train)**2)

plt.plot(J_hist)
plt.show()

# Evaluating Model using RMSE (root mean squared error)
predictions = np.dot(X_test,theta)
error = predictions - y_test
n = y_test[:,0].shape
RMSE = np.sqrt((sum(error)**2)/n)
print("Root Mean Square Error is:", RMSE[0])
