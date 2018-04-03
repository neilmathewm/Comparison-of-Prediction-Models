import numpy as np
import math
import scipy.stats as stats

'''
Module for misc. statistical methods 
for time series data
'''

# Detrends the target values of a time series
# by subtracting the residuals resulting
# from linear regression. 
# Returns a 3-tuple
# containing the vector of the detrended time series,
# and the slope and intercept of the regression line
def detrend(x,y):
	result = stats.linregress(x,y)
	slope = result[0]
	intercept = result[1]
	detrended = []
	for val in range(len(y)):
		pred = slope*val+intercept
		residual = y[val]-pred
		detrended.append(residual)
	return (detrended,slope,intercept)

# Applies a linear trend to a series of data points
# i.e. a[z] = y[z] + trend
#           = y[z] + slope*x[z] + intercept
def reapplyTrend(x,y,slope,intercept):
	rev = np.zeros(len(y))
	for dex in range(len(y)):
		rev[dex] = y[dex] + (slope*x[dex]+intercept)
	return rev

# Performs linear regression on univariate time
# series repr. by xTrain,yTrain, and then makes predictions
# for xTest
# Returns a 1-d vector representing predictions
def predictLinearRegression(xTrain,yTrain,xTest):
	result = stats.linregress(xTrain,yTrain)
	slope = result[0]
	intercept = result[1]
	predict = np.zeros(len(xTest))
	for x in range(len(xTest)):
		predict[x] = slope*xTest[x]+intercept
	return predict

# Performs linear regression on univariate time
# series repr. by xTrain,yTrain, and then makes predictions
# for xTest
# Returns a 1-d vector representing predictions
def predictPolyRegression(xTrain,yTrain,xTest,deg):
	coeff = np.polyfit(xTrain,yTrain,deg)
	predict = np.zeros(len(xTest))
	for x in range(len(xTest)):
		for c in range(len(coeff)):
			predict[x] += coeff[c]*math.pow(xTest[x],len(coeff)-c-1)
	return predict

# Computes the Mean Squared Error for predicted values against
# actual values
def meanSquareError(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	total = 0.0
	for x in range(len(actual)):
		total += math.pow(actual[x]-pred[x],2)
	return total/len(actual)

# Computes Normalized Root Mean Square Error (NRMSE) for
# predicted values against actual values
def normRmse(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	sumSquares = 0.0
	maxY = actual[0]
	minY = actual[0]
	for x in range(len(actual)):
		sumSquares += math.pow(pred[x]-actual[x],2.0)
		maxY = max(maxY,actual[x])
		minY = min(minY,actual[x])
	return math.sqrt(sumSquares/len(actual))/(maxY-minY)

# Computes Mean Absolute Percent Error (MAPE) for predicted
# values against actual values
def mape(actual,pred):
	if (not len(actual) == len(pred) or len(actual) == 0):
		return -1.0
	total = 0.0
	for x in range(len(actual)):
		total += abs((actual[x]-pred[x])/actual[x])
	return total/len(actual)

# Estimates missing data denoted by <targ> in
# a list of lists <data> by averaging the neighboring
# values if they exist, otherwise the mean of the entire
# list (i.e. if a missing value is at the start or end of
# a list)
def estimateMissing(data,targ):
    for x in range(len(data)):
        for y in range(len(data[x])):
            if (data[x][y] == targ):
                if (y > 0 and y < len(data[x])-1):
                    data[x][y] = (data[x][y-1]+data[x][y+1])/2.0
                else:
                    data[x][y] = np.mean(data[x])