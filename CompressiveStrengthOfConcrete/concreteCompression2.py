# 1 Import libaries and modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, \
RandomForestRegressor, GradientBoostingRegressor


##############################################################################
# MODEL
def model(algo, X_train, X_test, y_train, y_test, of_type):
  print('algorithm used:', algo)
  
  # fit to train data
  algo.fit(X_train, y_train)
  # predict for X_test
  prediction = algo.predict(X_test)
  # compare predictions against actual, y_test
  print('\nRMSE:', np.sqrt(mean_squared_error(y_test,prediction)))
  
  cross_val = cross_val_score(algo, X_train, y_train, cv=20, scoring='neg_mean_squared_error')
  cross_val = cross_val.ravel()
  print('cross validation scores:')
  print('cv_mean:',cross_val.mean())
  print('cv_std :',cross_val.std())
  print('cv_max :',cross_val.max())
  print('cv_min :',cross_val.min())
##############################################################################


# 2 Load Dataset
file = "E:\Datasets\Regression\ConcreteCompression\Concrete_Data.xls"
dataset = pd.read_excel(file)
# feature names are annoyingly long, going to abreviate them
dataset.columns = ["cement","furnace_slag",
                   "fly_ash","water","super_plasticizer",
                   "coarse_agg","fine_agg","age",
                   "compressive_str"]

# 3 Visualise dataset
# check first few rows
print('first 5 rows of dataset:\n',dataset.head(5))

# check for missing values
print('total number of missing values:')
print(dataset.isnull().sum())

# checking dimensions
(m,n) = dataset.shape
print('\ndimension of dataset:',dataset.shape)

# HISTOGRAM
for i in range(len(dataset.columns)):
  fig, ax = plt.subplots()
  ax.set_title(dataset.columns[i])
  ax.hist(dataset[dataset.columns[i]], bins='auto', alpha=0.7, rwidth=0.85, color='#2E8B57', density=True)
  dataset[dataset.columns[i]].plot.kde(ax=ax, color='#006400')
  plt.show()

# PAIRPLOT
pairplot = sns.pairplot(dataset)
for i,j in zip(*np.triu_indices_from(pairplot.axes,1)):
  pairplot.axes[i,j].set_visible(False)

# HEATMAP
cor = dataset.corr()
mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,10))
with sns.axes_style("white"):
    sns.heatmap(cor,annot=True,linewidth=2,
                mask = mask,cmap="magma")
plt.title("Correlation between variables")
plt.show()

# --------------------------------------------------------------------------
# Modelling

# splitting into training and testing datasets
X = dataset.iloc[:,0:n-1]
y = dataset['compressive_str']  
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                          test_size=0.2,
                                          random_state=7)

# running through different regression models
LINEAR = [LinearRegression(),Ridge(), Lasso()]
ENSEMBLE = [AdaBoostRegressor(),ExtraTreesRegressor(),RandomForestRegressor(),GradientBoostingRegressor()]
KNN = [KNeighborsRegressor()]

for i in LINEAR:
  model(i, X_train, X_test, y_train, y_test, 'coef')
  print('---------------------------------')
for i in ENSEMBLE:
  model(i, X_train, X_test, y_train, y_test, 'feat')
  print('---------------------------------')
for i in KNN:
#  knn = KNeighborsRegressor(n_neighbors=5, leaf_size=30,)
  model(i, X_train, X_test, y_train, y_test,'of_type')
  print('---------------------------------')

print('END......')
