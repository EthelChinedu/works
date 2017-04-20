#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:37:15 2017

@author: ethels
"""

#### Detrend a time series using differencing
#from pandas import read_csv
#from pandas import datetime
#from matplotlib import pyplot
#
#def parser(x):
#    return datetime.strptime('190'+x, '%Y-%m')
#
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0,
#                  squeeze=True, date_parser=parser)
#X = series.values
#diff = list()
#for i in range(1, len(X)):
#    value = X[i] - X[i -1]
#    diff.append(value)
#pyplot.plot(diff)
#pyplot.show()

#plots b4 detrending using differencing#
#from pandas import Series
#from matplotlib import pyplot
#series = Series.from_csv('shampoo-sales.csv', header=0)
#series.plot()
#pyplot.show()

#using linear model fit to detrend time series
#from pandas import read_csv
#from pandas import datetime
#from sklearn.linear_model import LinearRegression
#from matplotlib import pyplot
#import numpy 
#
#def parser(x):
#    return datetime.strptime('190'+x, '%Y-%m')
#
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0,
#                  squeeze=True, date_parser=parser)
#
##fit linear regressor 
#X = [i for i in range(0, len(series))]
#X = numpy.reshape(X, (len(X), 1))
#y = series.values
#model = LinearRegression()
#model.fit(X, y)
##cal trend
#trend = model.predict(X)
##plot trend
#pyplot.plot(y)
#pyplot.plot(trend)
#pyplot.show()
##plot detrend
#detrended = [y[i]-trend[i] for i in range(0, len(series))]
#pyplot.plot(detrended)
#pyplot.show()

#Deseasonalised time series using differecing

#from pandas import Series
#from matplotlib import pyplot
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#X = series.values
#diff = list()
#days_in_year = 365
#for i in range(days_in_year, len(X)):
#    value = X[i] - X[i - days_in_year]
#    diff.append(value)
#pyplot.plot(diff)
#pyplot.show

#model seasonality with a polynomial model
from pandas import Series
from matplotlib import pyplot
from numpy import polyfit
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#fit polynomial: X^2*b1 + X*b2 + ... + bn
X = [i%365 for i in range(0, len(series))]
y = series.values
degree = 4
coef = polyfit(X, y, degree)
#print('Coefficients: %s' % coef)
#create curve
curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)
#Create seasonaly adjusted to version of the dataset by subtracting the 
##values predicted by the seasonal model from the original obs
values = series.values
diff = list()
for i in range(len(values)):
    value = values[i] - curve[i]
    diff.append(value)
pyplot.plot(diff)
pyplot.show()
#plot curve over original data
#pyplot.plot(series.values)
#pyplot.plot(curve, color='red', linewidth=3)
#pyplot.show()