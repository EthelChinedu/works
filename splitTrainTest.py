#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:24:33 2017

@author: ethels
"""
##split into Train and Test Sets and testing a regressions
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values #when discussing BD, we forget abot that the is an array thou I've forgoten what its called
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#    random_state=seed)
#model = LogisticRegression()
#model.fit(X_train, Y_train)
#results = model.score(X_test, Y_test)
#print(results.mean())


# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = LinearDiscriminantAnalysis()
#results = cross_val_score(model, X, Y, cv=kfold)
#print(results.mean(), results.std())

#Evaluate with LDA
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
filename = 'Houzing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
         'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
#model = LinearRegression() : trying out on LASSORegress
#model = Lasso() : Trying out elasticNet
model = ElasticNet()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
print(dataframe.head(5))
