# -*- coding: utf-8 -*-
"""
@author: heman
"""
import pandas as pd
import numpy as np
#Function to convert categorical data (such as season) into string
def convert_categorical(data):
    data2 = data.copy()
    catcols = data2.columns[0:8]
    for col in catcols:
        data2[col] = data2[col].astype(str)
    return(data2)
#Take log of y variables
def y_log(df):
    data = df.copy()
    data['log_casual'] = np.log(data['casual'] + 1)
    data['log_registered'] = np.log(data['registered'] + 1)
    return(data)
#Function to engineer peak_registered and peak_casual features
def get_peaks(df):
    data = df.copy()
    data['peak_registered'] = 0
    data['peak_casual'] = 0
    data.loc[(data['workingday']==1) &((data['hr']==8) | ((data['hr']>=17)& (data['hr']<=18))),'peak_registered'] = 1
    data.loc[(data['workingday']==0) &((data['hr']>=11)& (data['hr']<=18)),'peak_registered'] = 1
    return(data)
#This function will be called from train.py and prediction.py
#It processes the data before training and prediction        
def get_features(df, dummies = True):
    data = df.copy()
    day = data['dteday']
    data.drop(['instant','dteday'], inplace = True, axis = 1)
    data = get_peaks(data)
    if dummies == True:
        data = convert_categorical(data)
        data = pd.get_dummies(data)
    data['dteday'] = day
    return(data)
#XY Split, should be handled differently for train as we are log-transforming on the training dataset    
def xySplit(df, kind = 'train'):
    data = df.copy()
    data.drop(['dteday'], axis = 1, inplace = True)
    if kind == 'train':
        data = y_log(data)
        y_casual = data['log_casual'] 
        y_registered = data['log_registered'] 
        X = data.drop(['log_casual','log_registered', 'casual','registered', 'cnt'], axis = 1)
    
    else:
        y_casual = data['casual']
        y_registered = data['registered'] 
        X = data.drop(['casual','registered', 'cnt'], axis = 1)
        
    return(X,y_casual,y_registered)
    