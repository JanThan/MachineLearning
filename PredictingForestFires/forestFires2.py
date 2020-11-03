import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
######################
# 0 Load Data
######################
file = 'forestfires.csv'
dataset = pd.read_csv(file)


######################
# 1 Explore Dataset
######################

(m, n) = dataset.shape
print('(Row, Column) =', dataset.shape)
print('------------------')

print(dataset.head(3))
print('------------------')

for i in range(n): # for all columns
  if i in [2,3]:
    pass
  else:
    print(i)
    plt.hist(dataset.iloc[:,i], alpha=0.7, rwidth=0.85, color='#2E8B57', density=True)
    plt.title(dataset.columns[i])
    plt.show()


######################
# 2 Data Preprocessing
######################
"""
deleting day from dataset as it assumed to be of little importance to 
predicting fires.
deleting rain from dataset as it is incredibly skewed.
"""
del dataset['day']
del dataset['rain']

"""
get dummy data for months
"""
dataset = pd.get_dummies(dataset)
print('Dummy data created...')
# updating m and n for dummy data
(m,n) = dataset.shape

# Extract feature and target variables from dataset
area_index = [i for i in range(0, n) if (dataset.columns[i] == 'area')]
non_area_index = [i for i in range(0, n) if (dataset.columns[i] != 'area')]  

"""
due to very skewed area data, making this a logistic regression problem:
--> will or won't there be a forest fire 
"""
X = dataset.iloc[:,non_area_index].values
y = np.zeros(X.shape[0])
for i in range(m):
  if dataset['area'][i] == 0:
    y[i] = 0
  else:
    y[i] = 1

# splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                   random_state = 7)
non_month_index = X.shape[1]-12
ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train[:, 0:non_month_index])
X_test_ss = ss.transform(X_test[:,0:non_month_index])
# re-introduce month information
X_train_ss = np.hstack((X_train_ss,X_train[:,non_month_index:]))
X_test_ss = np.hstack((X_test_ss, X_test[:,non_month_index:]))
print('Training and Test data created and standardised...')

######################
# 3 Feature Importance
######################

# using random forests
forest = RandomForestClassifier(n_estimators = 10000, random_state = 7)
forest.fit(X_train_ss, y_train)
importances = forest.feature_importances_
feature_labels = dataset.columns[non_area_index]
indices = np.argsort(importances)[::-1] # all items reversed

# plot feature importances
plt.figure(figsize=(12,10))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='#2E8B57', align='center')
plt.xticks(range(X_train.shape[1]), feature_labels[indices])
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# feature importance via principal component analysis
cov_mat = np.cov(X_train_ss[:, :non_month_index].T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('------------------')
print('eigen values:')
print(eigen_vals)
print('------------------')

######################
# 4 Modelling
######################
# logistic regression 
lr = LogisticRegression()
fitted_model = lr.fit(X_train_ss, y_train)
print('x test shape',X_test.shape)
y_pred = lr.predict(X_test)
print('Linear Regression:')
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))
print('------------------')

######################
# 5 Optimisation
######################
"""
from feature importance it can be seen that the months feature had 
little impact in comparison to other features. So now will fit a model 
"""
X_train_opt = X_train_ss[:,0:10]
X_test_opt = X_test_ss[:,0:10]
y_train_opt = y_train
y_test_opt = y_test
# logistic regression 
lr_opt = LogisticRegression()
fitted_model = lr_opt.fit(X_train_opt, y_train_opt)
print('x test shape',X_test_opt.shape)
y_pred_opt = lr_opt.predict(X_test_opt)
print('Linear Regression after removing months:')
print('Accuracy Score : ' + str(accuracy_score(y_test_opt,y_pred_opt)))
print('Precision Score : ' + str(precision_score(y_test_opt,y_pred_opt)))
print('Recall Score : ' + str(recall_score(y_test_opt,y_pred_opt)))
print('F1 Score : ' + str(f1_score(y_test_opt,y_pred_opt)))
print('------------------')

"""
ignoring months seemed to have little or worse effect on model 
"""