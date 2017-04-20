#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 07:45:26 2017

@author: ethels
"""
from pandas import Series
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from math import log
from math import sqrt
from math import exp
from sklearn.metrics import mean_squared_error
def boxcox_inverse(value, lam):
    if lam == 0:
        return exp(value)
    return exp(log(lam * value + 1) / lam)

series = Series.from_csv('dataset.csv')
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
#walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    tranformed, lam = boxcox(history)
    if lam < -5:
        transformed, lam = history, 1
    #predict
    model = ARIMA(tranformed, order=(0,1,2))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    #implemet the inverse transformed
    yhat = boxcox_inverse(yhat, lam)
    predictions.append(yhat)
    #observation
    obs = test[i]
    print(' >Predicted=%.3f, Expected=%3.f ' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)