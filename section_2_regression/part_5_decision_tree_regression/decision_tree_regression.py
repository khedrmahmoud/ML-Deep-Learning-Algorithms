"""
@author: Khedr
"""  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")

x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

#%% Fitting the Regression Model 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

#%% Predicting a new result
y_pred = regressor.predict(np.array(6.5).reshape(-1,1))

#%% Visualization
x_grid = np.arrange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color='r')
plt.plot(x_grid, regressor.predict(x_grid),color='g')
plt.title('(Decision Tree Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()