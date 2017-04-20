#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:42:48 2017

@author: ethels
"""
This is having issues, you have to verify

import pandas as pd
from matplotlib import pyplot
#filename = 'ShearTrng.csv'
#shearTr_df = pd.read_csv('ShearTrng.csv', header=1)
names = (['H59', 'H51', 'H30', 'H10', 'Mean59', 'SD59', 'Max59', 'Min59', 'Mean51', 
      'SD51', 'MeanWD', 'SDWD', 'MaxWD', 'MinWD', 'MeanTemp', 'SDTemp', 'MaxTemp',
     'MinTemp', 'Date', 'Time'])
filename = 'ShearTrng.csv'
dataset = pd.read_csv(filename, names=names)
#Summerise the dataset by checking the 
#Dimension, peeking at the datat itself, do summary stats and have a breakdown
##of the class variable
print(dataset.shape)
print(dataset.head(3))
#print(dataset.describe())
#print(shearTr_df.head(4))
MeanWindsp = dataset[['Date', 'Mean59', 'Max59', 'Min59', 'SD59']]
MeanWindsp2 = dataset[['Date', 'Mean51', 'Min51', 'SD51']]
PlotSD49_39 = dataset[['Date', 'SD59', 'SD51']]
#print(dataset.tail(4))
MeanWindsp.plot()
MeanWindsp2.plot()
PlotSD49_39.plot()
pyplot.show()