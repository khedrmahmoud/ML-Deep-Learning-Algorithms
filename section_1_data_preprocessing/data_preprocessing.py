"""
@author: Khedr
"""  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Data.csv")

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

#%% Taking care of missing data

from sklearn.impute import SimpleImputer
imputer =SimpleImputer(missing_values=np.nan,strategy="mean")
imputer=imputer.fit(x[:,1:3])

x[:,1:3]= imputer.transform(x[:,1:3])

#%%
#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X = LabelEncoder()
x[:,0]=labelEncoder_X.fit_transform(x[:,0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
x=ct.fit_transform(x)

labelEncoder_Y = LabelEncoder()
y=labelEncoder_X.fit_transform(y)

#%% Spliting data into training and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


#%% Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train[:,3:] = sc_x.fit_transform(x_train[:,3:])
x_test[:,3:] = sc_x.transform(x_test[:,3:])

 















