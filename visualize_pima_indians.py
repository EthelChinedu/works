#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:10:30 2017

@author: ethels
"""
# Univariate Histograms
#from matplotlib import pyplot
#from pandas import read_csv
#import numpy
#filename = 'pima-indians-diabetes.data.csv'
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#data = read_csv(filename, names=names)
#correlations = data.corr()
##plot correlation matrix
#fig = pyplot.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(correlations, vmin=-1, vmax=1)
#fig.colorbar(cax)
#ticks = numpy.arange(0,9,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
#data.hist()
#pyplot.show()

#Demonstrate Data Rescaling and normilisations
#from matplotlib import pyplot
from numpy import set_printoptions
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values #when discussing BD, we forget that the is an array thou I've forgoten what its called
X = array[:,0:8]
Y = array[:,8]
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler().fit(X)
rescaledX = scaler.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])
print(dataframe.head(5))
print(X.head(5))


