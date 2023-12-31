# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Get the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%% Encoding Categorical data
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X = LabelEncoder()
x[:,3]=labelEncoder_X.fit_transform(x[:,3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
x=ct.fit_transform(x)

#%% Avoiding dumy variable trap
x=x[:,1:]

#%% Spiliting the dataset into the training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0)


#%% Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

#%% Predicting the Test set results
y_pred=regressor.predict(X_test)


#%% Building the optimal model using Backward Elimination
import statsmodels.api as sm

x=np.append(arr= np.ones(shape=(50,1)), values=x,axis=1)
X_opt=x
regressor_OLS=sm.OLS(endog= y ,exog=X_opt ,hasconst=False).fit(method="qr")
regressor_OLS.summary()

#%%
X_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog= y ,exog=X_opt ,hasconst=False).fit(method="qr")
regressor_OLS.summary()

#%%
X_opt=x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog= y ,exog=X_opt ,hasconst=False).fit(method="qr")
regressor_OLS.summary()

#%%
X_opt=x[:,[0,3,5]]
regressor_OLS=sm.OLS(endog= y ,exog=X_opt ,hasconst=False).fit(method="qr")
regressor_OLS.summary()
#%%
X_opt=x[:,[0,3]]
regressor_OLS=sm.OLS(endog= y ,exog=X_opt ,hasconst=False).fit(method="qr")
regressor_OLS.summary()


#%%Backward Elimination with p-values only:

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#%%Backward Elimination with p-values and Adjusted R Squared:

def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)







