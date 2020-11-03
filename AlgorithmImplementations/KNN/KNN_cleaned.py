# Implementing KNN from scratch
import numpy as np
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn import model_selection
import matplotlib.pyplot as plt
"""
Method:
  - generate a dataset of 'coordinates' (x1, x2), each with a corresponding 
class (y value)
  - split this data into two sets, training and test data
  - the actual y value for the test data will not matter so much as there is no 
pre-existing grouping here to compare to. In an actual implementation, we would 
want to compare the predicted y value to the actual to get a sense of accuracy.
As a result, for this implementation, I will be relying on visual validation 
  - I will assign each of the test data points with a class depending on the kNN
  - Then consequently visualise all the data

"""


def Edist(p1,p2):
  # calculates the euclidean distance between p1 and p2
  dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
  return dist

def sortAndCount(array, k, train_number, test_number):
  # INPUT: the distance array (last column is y_train), k, and number of 
  # test and train examples
  # OUTPUT: the class of each test example after finding its KNN
  
  # array --> rows-train, columns-test
  outputy = array[:,-1]
  inputx = array[:,0:test_number]
  testPoints = np.zeros(test_number)
  
  for i in range(test_number):
    # for each test, sort in distances and then find max class for k closest
    workingArray = np.column_stack((inputx[:,i],outputy)) # get distances for the i-th test
    sortedArray = workingArray[np.argsort(workingArray[:,0])] # sort for the 1st column (distances)
    # use 'Counter' to find which class (0 or 1 here) appears the most
    testPoints[i] = Counter(sortedArray[0:k,1]).most_common(1)[0][0]
  return testPoints
     
def inputNumber(message):
  while True:
    try:
       userInput = int(input(message))       
    except ValueError:
       print("Not an integer! Try again.")
       continue
    else:
       return userInput 
       break 
     

# Generate a random dataset to use
x1 = pd.Series(np.random.normal(1,2,50))
x2 = pd.Series(np.random.normal(1,2,50))
X = pd.concat([x1, x2], axis=1, keys=['x1','x2']) # combining coordinates
y = pd.Series(np.repeat([0,1],25))

# seperate into 'test' and train datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,
                                      test_size = 0.2, random_state = 7)
# store number of examples in each set
m_train = (X_train.shape)[0]
m_test = (X_test.shape)[0]

distance = np.zeros((m_train, m_test)) # rows-train, columns-test
#k = 3 # initialising K value, keep this as an odd number to avoid conflicts
k = inputNumber("Enter Value of K:")

# converting to numpy arrays for easier indexing IMO
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
# bit of extra fiddling was needed to get y_train to the right size
y_train = np.transpose(np.array([y_train.to_numpy()])) 

# for each test point, find and store e-distance from all existing training values
for j in range(m_test):
  for i in range(m_train):
    p1 = X_train[i,:]
    p2 = X_test[j,:]
    distance[i,j] = Edist(p1,p2) 

# combining to one array with the corresponsing y values for x_train values
distance_with_y = np.hstack((distance, y_train))
# send merged array to find KNN for each test point
classified_test_results = sortAndCount(distance_with_y, k, m_train, m_test)

trainingData = np.hstack((X_train, y_train))
trainingData = pd.DataFrame({'x1':trainingData[:,0],'x2':trainingData[:,1],'y':trainingData[:,2]})

resultsData = np.column_stack((X_test,classified_test_results))
resultsData = pd.DataFrame({'x1':resultsData[:,0],'x2':resultsData[:,1],'y':resultsData[:,2]})

# SPLITTING DATA by y classes just so I can visualise everything on the same 
# scatter plot below

resultsData = resultsData.to_numpy()
trainingData = trainingData.to_numpy()

zeroArrayR = []
oneArrayR = []
zeroArrayT = []
oneArrayT = []

# Results Data
for i in range(len(resultsData[:,2])):
  if resultsData[i,2] == 0:
    zeroArrayR.append(resultsData[i,:])
  elif resultsData[i,2] == 1:
    oneArrayR.append(resultsData[i,:])
  else:
    print('invalid element in y column')

# Training Data
for i in range(len(trainingData[:,2])):
  if trainingData[i,2] == 0:
    zeroArrayT.append(trainingData[i,:])
  elif trainingData[i,2] == 1:
    oneArrayT.append(trainingData[i,:])
  else:
    print('invalid element in y column')

# converting again to numpy arrays for easier indexing  
zeroArrayR = np.array(zeroArrayR)
oneArrayR = np.array(oneArrayR)
zeroArrayT = np.array(zeroArrayT)
oneArrayT = np.array(oneArrayT)

# visualising data
zt = plt.scatter(zeroArrayT[:,0],zeroArrayT[:,1], c='blue', marker='o')
ot = plt.scatter(oneArrayT[:,0],oneArrayT[:,1], c='orange', marker='o')
zp = plt.scatter(zeroArrayR[:,0],zeroArrayR[:,1], c='blue', marker='x')
op = plt.scatter(oneArrayR[:,0],oneArrayR[:,1], c='orange', marker='x')

plt.legend((zt, zp, ot, op),
           ('Training 0s', 'Predicted 0s', 'Training 1s', 'Predicted 1s'),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8, bbox_to_anchor=(-0.1, 1.2))
plt.show()
