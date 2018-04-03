import numpy as np
from sklearn import linear_model
from sklearn import svm
import pandas as pd
import quandl,math
import numpy as np
import statistics
import xlrd

  

xData=[]
yData=[]
print "Fetching Data"
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
#print (xData)
#print (yData)
cu=len(xData)-720
cutoff = len(xData)-30
print(cutoff)
xTrain = xData[cu:cutoff]
  
  #print xTrain[47]
  #print xTrain
yTrain = yData[cu:cutoff]
xTest = xData[cutoff:]
  #print cutoff
  #print xTest[0]
yTest = yData[cutoff:]
#print (yTest)

classifiers = [
    svm.SVR(kernel='linear'),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]



for item in classifiers:
    print "------------------------------------------------------------------"
    print(item)
    clf = item
    clf.fit(xTrain,yTrain)
    pred=clf.predict(xTest)
    print(pred)
    print(yTest)
    err2=statistics.mape(pred,yTest)
    print("Error Rate :"+str(err2)+"\n")
    print("% Accuracy :"+str((1-err2)*100)+"\n")
    print "------------------------------------------------------------------"

