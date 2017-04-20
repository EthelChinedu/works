#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:25:12 2017

@author: ethels
"""
##line plot in python
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
series.plot()
pyplot.show()

##changing the style of the plot to a dot plots

#from pandas import Series
#from matplotlib import pyplot
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#series.plot(style='b.')
#pyplot.show()

#create a histogram of the data

#from pandas import Series
#from matplotlib import pyplot
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#series.hist()
#pyplot.show()

#create a density plot of the same dataset

#
#from pandas import Series
#from matplotlib import pyplot
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#series.plot(kind='kde')
#pyplot.show()

####comparing for the same interval s.a d-t-d, m-t-m or even y-t-y for like 
#####10 yrs interval. Here first obs are grouped by year using A. 
#furthermore the group are then enumerated and the obs for each year are stored
###as col in a new dataFrame . then a plot of this dataFrame is created with each col,
###visualised as a sub-plot with legends removed to cut back on the cluther.

#from pandas import Series
#from pandas import DataFrame
#from pandas import TimeGrouper
#from matplotlib import pyplot
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#groups = series.groupby(TimeGrouper('A'))
#years = DataFrame()
#for name, group in groups:
#    years[name.year] = group.values
#years.plot(subplots=True, legend=False)
##series.plot() using this line will plot the series in each sub-plot
#pyplot.show()

#creating boxplot and whiskers
#
#from pandas import Series
#from pandas import DataFrame
#from pandas import TimeGrouper
#from matplotlib import pyplot
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
#groups = series.groupby(TimeGrouper('A'))
#years = DataFrame()
#for name, group in groups:
#    years[name.year] = group.values
#years.boxplot()
##series.plot() using this line will plot the series in each sub-plot
#pyplot.show()

#create an autocorrelation plot 

from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
autocorrelation_plot(series)
pyplot.show()

