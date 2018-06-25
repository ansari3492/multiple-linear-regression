# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:53:58 2018

@author: Lenovo
"""

import pandas as pd
import numpy as np
data=pd.read_csv("stats_females.csv")
features=data.iloc[:,1:]
labels=data["Height"].values



#splitting the data into train and test
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


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
features=np.append(arr=np.ones((214,1)).astype(int) ,values=features,axis=1)

#optimize the solution
features_opt=features[:,[0,1,2]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()

#optimize the solution
features_opt=features[:,[1,2]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()
coef=reg_ols.params

print("Dad constant",coef[0])


print("Mom constant",coef[1])