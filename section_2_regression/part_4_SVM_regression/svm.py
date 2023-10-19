# =============================================================================
# SVR
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")

x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,-1].values
#%% Spliting data into training and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#%% Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y =np.ravel( sc_y.fit_transform(y.reshape(-1,1)))
#%% Fitting SVM regression 
from sklearn.svm import SVR

regessor=SVR(kernel= 'rbf')
regessor.fit(x, y)

#%% Visualising the polynomial regresion
x_gred= np.arange(min(x),max(x)+0.1,0.1)
x_gred  = x_gred.reshape(len(x_gred),1)

plt.scatter(x, y, color='r')
plt.plot(x_gred,regessor.predict(x_gred), color='b')
plt.title("Truth or Bluff(SVM Regression)")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()

#%% Predicting a new result with linear regression
y_pred=regessor.predict(sc_x.transform(np.array(6.5).reshape(1, -1)))
y_pred= sc_y.inverse_transform(y_pred.reshape(1, -1))
