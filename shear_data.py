#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:17:49 2017

@author: ethels
"""
from pandas import read_csv
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import datetime
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot
shear_df = read_csv('shear1.csv', header=0)
#shear_df = pd.read_csv('shear1.csv', header=0, parse_dates=[0], index_col=0,
 #                      squeeze=True)#', header=0)
#names = (['H49', 'H39', 'H30', 'H10', 'Mean49', 'SD49', 'Max49', 'Min49', 'Mean39', 
#       'SD39', 'MeanWD', 'SDWD', 'MaxWD', 'MinWD', 'MeanTemp', 'SDTemp', 'MaxTemp',
#      'MinTemp', 'Date', 'Time'])
#shear1.csv.query('H49 > Time')
#shear1.csv.query('H49 == Time')
#pd.isnull('shear1.csv')
#mean = np.mean(shear_df)
#print(mean.head(4))
#df.interpolate(shear_df)
#print(shear_df.isnull().sum())
usable = shear_df[['Mean49', 'SD49', 'Max49', 'Min49', 'Mean39', 'SD39', 'MeanWD', 'SDWD', 'MaxWD', 'MinWD', 'MeanTemp', 'SDTemp', 'MaxTemp',
     'MinTemp', 'Date', 'Time']]
MWsp = shear_df[['Date', 'Mean49', 'Max49', 'Min49', 'SD49']]
MWdsp2 = shear_df[['Date', 'Mean39', 'Min39', 'SD39']]
SD = shear_df[['Date', 'SD49', 'SD39']]
series = shear_df[['SD30', 'Date']]
series.to_csv('series.csv')
seriz = Series.from_csv('series.csv')
seriz = DataFrame(series)
#seriz = seriz.values.astype('float64')
#print(shear_df.isnull())
#MWarray = np.asarray(MWwind)
#print(series.tail(10))
print(usable.Mean49.describe())
print((usable.Mean49 <= 10.96) | (usable.MeanWD == '200')])
#print(usable.head(10))
#pyplot.figure()
#pyplot.subplot(211)
#usable.Mean49.plot(kind='kde', ax=pyplot.gca())
#pyplot.subplot(212)
#usable.SD49.plot(kind='kde', ax=pyplot.gca())
#pyplot.subplot(211)
#usable.Max49.plot(kind='kde', ax=pyplot.gca())
#pyplot.subplot(211)
#usable.Min49.plot(kind='kde', ax=pyplot.gca())
#pyplot.show()


#seriz.plot()
#MWsp.plot()
#MWdsp2.plot()
#SD.plot()
#pyplot.show()
#autocorrelation_plot(seriz)
#plot_acf(seriz)
#print(shear_df.size)
#model = ARIMA(MWwind, order=(5,1,0))
#model_fit = model.fit(disp=0)
##Do summery of the fit model
#print(model_fit.summary())
#residuals = DataFrame(model_fit.resid)
#residuals.plot()
#residuals.plot(kind='kde')
#pyplot.show()
#print(residuals.describe())