# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:07:03 2018

@author: Lenovo
"""

import pandas as pd
import numpy as np
data=pd.read_csv("iq_size.csv")
features=data.iloc[:,1:]
labels=data["PIQ"].values

#splitting the data into train and test
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


#feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)


#linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(features_train,labels_train)


#predict the results
pred=reg.predict(features_test)

#score results
score=reg.score(features_train,labels_train)

#building the optimal model backward elimination
import statsmodels.formula.api as sm
features=np.append(arr=np.ones((38,1)).astype(int) ,values=features,axis=1)

#optimize the solution
features_opt=features[:,[1,2,3]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()


#optimize the solution
features_opt=features[:,[1,2]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()

#optimize the solution
features_opt=features[:,[1]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()
