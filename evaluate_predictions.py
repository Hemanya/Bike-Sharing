# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 07:20:09 2020

@author: heman
"""

import pandas as pd
import numpy as np

def evaluatePredictions(y_pred,y_true):
    mad = np.mean(np.absolute(y_pred - y_true))
    mape = np.mean(np.abs(np.absolute(y_true - y_pred) / y_true)) * 100
    return(mad,mape)
    

y_pred = pd.read_csv('y_pred.csv')
y_true = pd.read_csv('y_true.csv')

mad, mape = evaluatePredictions(np.array(y_pred),np.array(y_true))

print("MAD", mad)
print("MAPE",mape)