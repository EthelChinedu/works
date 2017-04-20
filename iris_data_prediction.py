#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:52:08 2017

@author: ethels
"""

from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Load Dataset
filename = 'iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)
#Summerise the dataset by checking the 
#Dimension, peeking at the datat itself, do summary stats and have a breakdown
##of the class variable
print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())
#Do a class distribution on the dataset to see how distributed the data is
print(dataset.groupby('class').size())
#creating box/whisker plot for univariate plts
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#create histograms
dataset.hist()
pyplot.show()
#create multivariate plots to see the interactions btw variables
scatter_matrix(dataset)
pyplot.show()
#model evaluation. Here we are going to separate out a validation dataset
#setup the test harness to use 10-fold cross-validation
#Build 5 diff models to predict species from flower measurements 
#Select the best model
#SPLIT OUT THE VALIDATION DATASET
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                    test_size=validation_size, random_state=seed)
#here you now have training data in the X_train and Y_train for preparing models 
#and X_validation and Y_validation set that can be used later
#After the validation, we build models
#bc we dont know the nest algorithm we are finally going to use, we'd do all
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#evaluate each model in turn
results =[]
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#Graphically compare algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
#ax.set_xticklabes(names)
pyplot.show()
# Make predictions on validation dataset
svm = SVC()
#knn = KNeighborsClassifier()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



