#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:52:18 2017

@author: ethels
"""
#creat date time feature of a dataset
from pandas import Series
from pandas import DataFrame
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
dataframe = DataFrame()
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].month for i in range(len(series))]
dataframe['temperature'] = [series[i] for i in range(len(series))]
print(dataframe.head(40))

####create linear window feature engineering
#from pandas import Series
#from pandas import DataFrame
#from pandas import concat
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#temps = DataFrame(series.values)
#dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
#dataframe.columns =['t-2', 't-1', 't', 't+1'] 
#print(dataframe.head(5))

##creating a rolling window feature engring
##create linear window feature engineering
#from pandas import Series
#from pandas import DataFrame
#from pandas import concat
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#temps = DataFrame(series.values)
#shifted = temps.shift(1)
#window = shifted.rolling(window=4)
#means = window.mean()
#dataframe = concat([means, temps], axis=1)
#dataframe.columns =['mean(t-3,t-2,t-1,t)', 't+1'] 
#print(dataframe.head(10))

##################################################
from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
temps = DataFrame(series.values)
#width = 3
shifted = temps.shift(3 - 1)
#window = shifted.rolling(window=3)
window = temps.expanding() #implementing expanding windo feature engineering
dataframe = concat([window.min(), window.mean(), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(10))