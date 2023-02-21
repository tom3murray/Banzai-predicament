# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 06:29:43 2023

@author: Lyle_
"""

import pandas as pd
import os
from sklearn import linear_model
import statsmodels.api as sm


os.chdir('C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS')
os.getcwd()
df = pd.read_excel('regression_data.xlsx')


# Source: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
one_hot = pd.get_dummies(df['pres_party'])
# Drop current column B 
df = df.drop('pres_party',axis = 1)
# Join the one_hot column
df = df.join(one_hot)


list(df.columns)

#%%
# CODE FOR MULTIPLE LINEAR REGRESSION 

# Source: https://datatofish.com/multiple-linear-regression-python/
x = df[['AVG troops as prop of pop',
        'recession',
        'age',
        'months_til_pres_election',
        'pres_house_pct_ctrl_bool',
        'pres_senate_pct_ctrl_bool',
        'word_count',
        'Democratic-Republican',
        'Federalist',
        'Post-FDR Democratic',
        'Post-FDR Republican',
        'Pre-FDR Democratic',
        'Pre-FDR Republican',
        'Whig']]
y = df['pos']
 
# # with sklearn
# regr = linear_model.LinearRegression()
# regr.fit(x, y)

# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)
