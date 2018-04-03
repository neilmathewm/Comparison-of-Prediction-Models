import datetime
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from scipy import signal

'''
Module for time series visualization
'''

# Plots the (x,y) data using matplotlib with the given labels
def yearlyPlot(ySeries,year,month,day,plotName ="Plot",yAxisName="yData"):

	date = datetime.date(year,month,day)
	dateList = []
	for x in range(len(ySeries)):
		dateList.append(date+datetime.timedelta(days=x))

	plt.plot_date(x=dateList,y=ySeries,fmt="r-")
	plt.title(plotName)
	plt.ylabel(yAxisName)
	plt.xlabel("Date")
	plt.grid(True)
	plt.show()

# Plots autocorrelation factors against varying time lags for ySeries
def autoCorrPlot(ySeries,plotName="plot"):
	plt.figure()
	plt.title(plotName)
	data = pandas.Series(ySeries)
	autocorrelation_plot(data)
	plt.show()

# Displays lag plot to determine whether time series data is non-random
def lagPlot(ySeries,plotName="plot"):
	plt.figure()
	plt.title(plotName)
	data = pandas.Series(ySeries)
	lag_plot(data, marker='2', color='green')
	plt.show()

# Displays periodogram of the given time series <ySeries>
def periodogramPlot(ySeries,plotName="Plot",xAxisName="Frequency",yAxisName="Frequency Strength"):
	trans = signal.periodogram(ySeries)
	plt.title(plotName)
	plt.xlabel(xAxisName)
	plt.ylabel(yAxisName)
	plt.plot(trans[0], trans[1], color='green')
	plt.show()

# Plots two time series on the same timeScale from a common date on the same plot
def comparisonPlot(year,month,day,seriesList,nameList,plotName="Comparison of Values over Time", yAxisName="Predicted"):
	date = datetime.date(year,month,day)
	dateList = []
	for x in range(len(seriesList[0])):
		dateList.append(date+datetime.timedelta(days=x))
	colors = ["b","g","r","c","m","y","k","w"]
	currColor = 0
	legendVars = []
	for i in range(len(seriesList)):
		x, = plt.plot_date(x=dateList,y=seriesList[i],color=colors[currColor],linestyle="-",marker=".")
		legendVars.append(x)
		currColor += 1
		if (currColor >= len(colors)):
			currColor = 0
	plt.legend(legendVars, nameList)
	plt.title(plotName)
	plt.ylabel(yAxisName)
	plt.xlabel("Date")
	plt.show()