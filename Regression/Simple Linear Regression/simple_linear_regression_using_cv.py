# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 19:14:44 2022

@author: Sisir.Sahu
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#sklearn gives us some of the base datasets to practice
from sklearn.datasets import load_boston
load_boston()

df = load_boston()

#creating a dataframe
data = pd.DataFrame(df.data)

#assigning col names to the dataframe 
data.columns = df.feature_names
data.head()

#Independent features and dependent features
x = data
y = df.target

type(x)
type(y)

#split data into train and test
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#since we have diff features with diff units, standard scaling features make sense
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
#in the above step, scaler has already learnt from fit_transform, we want to surprise it by by passing just transform in x test
x_test = scaler.transform(x_test)

#if we want to go back to our original values, we will do inverse transform
x_train_original = scaler.inverse_transform(x_train)
x_test_original = scaler.inverse_transform(x_test)

#importing linear regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
lr.fit(x_train_original, y_train)

#Since we have so many features, we will use Cross Validation
from sklearn.model_selection import cross_val_score
mse = cross_val_score(lr, x_train, y_train, scoring='neg_mean_squared_error', cv = 10)
mse_inv = cross_val_score(lr, x_train_original, y_train, scoring='neg_mean_squared_error', cv = 10)

mse_inv.mean()

y_pred = lr.predict(x_test)
y_pred_inv = lr.predict(x_test_original)

import seaborn as sns
sns.displot(y_pred - y_test, kind = 'kde')

from sklearn.metrics import r2_score
residuals = r2_score(y_test, y_pred)
residuals

