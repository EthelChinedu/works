#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:40:20 2017

@author: ethels
"""

from pandas import read_csv
series = read_csv('daily-total-female-births.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(type(series))
##print(series['1959-01']) query ur data by date-time
print(series.describe())

