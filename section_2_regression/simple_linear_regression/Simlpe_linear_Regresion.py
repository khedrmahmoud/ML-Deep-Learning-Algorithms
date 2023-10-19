"""
@author: Khedr 
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Get the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# %% Spiliting the dataset into the training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)


# %% fitting simple linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %% predecting the test set
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

# %% visualising the train set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_pred_train, color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()
# %% visualising the test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()
