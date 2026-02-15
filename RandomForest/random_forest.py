#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import scienceplots

plt.style.use(['science','no-latex'])

# ---- for k-fold cross-validation ----- 

def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (metrics.r2_score(y, yPred), 
            metrics.mean_absolute_error(y, yPred))

def my_scorer(estimator, x, y):
    a, p = getScores(estimator, x, y)
    print(a, p)
    return a + p

cv = KFold(n_splits=5, shuffle=True)

# --------------------------------------

datadf = pd.read_csv("/Users/admin/projects/AlN/ML MODELS/dataset.csv", na_values = 'not reported')
datadf = datadf.to_numpy()

targetdf = pd.read_csv("/Users/admin/projects/AlN/ML MODELS/target.csv")
targetdf = targetdf.to_numpy()
targetdf = targetdf.ravel()

imputer = MissForest() 
datadf = imputer.fit_transform(datadf)

x_train, x_test, y_train, y_test = train_test_split(datadf, targetdf, shuffle=True)

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

reg = RandomForestRegressor(n_estimators = 12000, random_state = 42)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print('\nR2: %f\nMAE: %f' % (metrics.r2_score(y_test, y_pred), metrics.mean_absolute_error(y_test, y_pred)))

plt.plot(y_test, y_test)
plt.xlabel('d33,f (pm/V)')
plt.ylabel('d33,f (pm/V)')
plt.scatter(y_test, y_pred)
plt.show()

importance = reg.feature_importances_

for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()