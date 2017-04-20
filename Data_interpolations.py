#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:09:05 2017

@author: ethels
"""
### how to interpolate data - either upsampling or downsampling
#from pandas import read_csv
#from pandas import datetime
#
#def parser(x):
#    return datetime.strptime('190'+x, '%Y-%m')
#
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0,
#               squeeze=True, date_parser=parser)
#upsampled = series.resample('D').mean()
#print(upsampled.head(32))

##with this code, we have missing values so what we do is to interpolate the missing values 
###there are many options but in our case we are using linear interpolation

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0,
               squeeze=True, date_parser=parser)
upsampled = series.resample('D').mean()
interpolated = upsampled.interpolate(method='spline', order=2)
print(interpolated.head(32))
interpolated.plot()
pyplot.show()
print(series.head(4))


