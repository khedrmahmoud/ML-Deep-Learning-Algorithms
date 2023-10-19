import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")

x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,-1].values


#%% Fitting Linear regression 
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(x, y)

#%% Fitting Polynomial regression
from sklearn.preprocessing import PolynomialFeatures

ploy_reg = PolynomialFeatures(degree=4)
x_poly = ploy_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#%% Visualising the linear regresion
plt.scatter(x, y, color='r')
plt.plot(x,lin_reg.predict(x), color='b')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()

#%% Visualising the polynomial regresion
x_gred= np.arange(min(x),max(x)+0.1,0.1)
x_gred  = x_gred.reshape(len(x_gred),1)

plt.scatter(x, y, color='r')
plt.plot(x_gred,lin_reg_2.predict(ploy_reg.fit_transform(x_gred)), color='b')
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()


#%% Predicting a new result with linear regression
lin_reg.predict(np.array(6.5).reshape(1, -1))


#%% Predicting a new result with polynomial regression

lin_reg_2.predict(ploy_reg.fit_transform(np.array(6.5).reshape(1, -1)))






