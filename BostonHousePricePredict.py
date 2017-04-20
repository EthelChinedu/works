#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:39:11 2017

@author: ethels
"""

#Boston house dataset prediction
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm  import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
#load dataset
filename = 'Houzing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
         'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)
print(dataset.shape)
print(dataset.dtypes)
set_option('precision', 1)
print(dataset.describe())
#Take a look at the correlation between the numeric attributes 
set_option('precision', 2)
print(dataset.corr(method='pearson'))
#Data Visualisation, first is by using histogram/density/whisker/etc
###dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
###pyplot.show()
#density plot
###dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, 
###             legend=False, fontsize=1)
#Box plot
###dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, 
###             legend=False, fontsize=8)
###pyplot.show()
#scatter plot matrix
###scatter_matrix(dataset)
###pyplot.show()
#VAlidation Dataset 
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20  #sample of the data to be held back from our analysis/modeling
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
            test_size=validation_size, random_state=seed)
#Test options and evaluation metric
num_fold = 10
scoring = 'neg_mean_squared_error'
#Spot-check algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append (('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_fold, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#print(cv_results)
#Algorithm tunning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1, 3, 5, 7, 9, 11, 13, 17, 17, 21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_fold, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
#to display mean n STD scores across the creadted folds
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
##since the tunning on KNN didnt improve the result, we will try ensemble technique
####ensembles using Adaboost/etxtraTree Regressor
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), 
                            ('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM',
                            GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF',
                            RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET',
                  ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_fold, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#compare algorithms
fig = pyplot.figure()
fig.suptitle( ' Scaled Ensemble Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
    


