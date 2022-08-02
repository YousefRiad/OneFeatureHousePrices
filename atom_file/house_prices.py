import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#getting the data
df = pd.read_csv('one_feature_house_prices.csv')


#plotting the data
%matplotlib inline
plt.xlabel('area(sqr ft)')
plt.ylabel('price($)')
plt.scatter(df.area,df.price,color='red', marker='x')

#usning linear reg
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


#making a predict
reg.predict([[3200]])

#w=
reg.coef_

#b=
reg.intercept_


%matplotlib inline
plt.xlabel('area(sqr ft)',fontsize=15)
plt.ylabel('price($)',fontsize=15)
plt.scatter(df.area,df.price,color='red', marker='x')
plt.plot(df.area,reg.predict(df[['area']]),color = 'blue')

testAreas = pd.read_csv('test_areas.csv')
testAreas

predictedPrices = reg.predict(testAreas)
predictedPrices

#to assign the predicted price values to the CSV file:
#1-create a new column with the header 'prices'
#2-assign the predicted values to that new column

# testAreas['prices'] = predictedPrices


#to send the predictions to a new CSV file:
testAreas['prices'] = predictedPrices
testAreas.to_csv("predictions.csv",index = False)
