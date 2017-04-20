#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:03:26 2017

@author: ethels
"""
from pandas import Series
from pandas.tools.plotting import autocorrelation_plot
series = Series.from_csv('daily-total-female-births.csv')
series.
#autocorrelation_plot(series)