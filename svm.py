import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
import statistics
import xlrd

  
  # Retrieve time series data & apply preprocessing
  #print tdata
  # 2014 had 365 days, but we take the last 364 days since
  # the last day has no numerical value
xData=[]
yData=[]
book = xlrd.open_workbook("data/data_with_9_variable.xlsx")
sheet = book.sheet_by_index(0)
for rx in range(1,sheet.nrows):
    row = sheet.row(rx)[1:12] #including temps
    rowy=sheet.row(rx)[12] #total of next day
    row = [row[x].value for x in range(0,len(row))]
    rowy=rowy.value
    xData.append(row)
    yData.append(rowy)
    #print "cutoff"+str(cutoff)
print (xData)
print (yData)
cu=len(xData)-720
cutoff = len(xData)-30
#print(cutoff)
xTrain = xData[cu:cutoff]
  
  #print xTrain[47]
  #print xTrain
yTrain = yData[cu:cutoff]
xTest = xData[cutoff:]
  #print cutoff
  #print xTest[0]
yTest = yData[cutoff:]
print (yTest)
  
print ("SVM")
for k in ['sigmoid','linear']:
    clf = svm.SVR(kernel=k)
    clf.fit(xTrain, yTrain)
    forecast_set = clf.predict(xTest)
    confidence = clf.score(xTest, yTest)
    print (forecast_set)

    err2=statistics.mape(forecast_set,yTest)
    print(k,"error rate:"+str(err2),"% Accuracy : "+str((1-err2)*100))

