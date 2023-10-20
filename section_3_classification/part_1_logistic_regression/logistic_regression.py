"""
@author: Khedr
"""  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

#%% Spliting data into training and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train  = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#%% Fitting the Classification Model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0 )
classifier.fit(x_train, y_train)

#%% Predicting the test set
y_pred = classifier.predict(x_test)

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Visualising the Trainging set result
from matplotlib.colors import ListedColormap

















