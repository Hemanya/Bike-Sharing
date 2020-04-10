# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 07:27:08 2020

@author: heman
"""
import pandas as pd
import numpy as np
import pickle
import process_data

#No need to call this function. This is only used to create test data. 
#In production, the data will be dynamic and be imported directly
def make_prediction_data(date):
    df = pd.read_csv('hour.csv')
    df = df[df['dteday'] >= date] #On and after Sep 1
    y = pd.DataFrame(df['cnt'], columns = ['cnt'])
    df = df.drop(['casual', 'registered','cnt'],axis = 1)
    
    df.to_csv('test_data.csv', index = False)
    y.to_csv('y_true.csv', index = False)

#Import models and return prediction    
def predict(testX):
    filename = 'casual_model.sav'
    casual_model = pickle.load(open(filename, 'rb'))
    
    filename = 'registered_model.sav'
    registered_model = pickle.load(open(filename, 'rb'))
    
    X_casual = testX.drop(['peak_registered'], axis = 1)
    X_registered = testX.drop(['peak_casual'], axis = 1)
    
    ypred_casual = np.round(np.exp(casual_model.predict(X_casual)) - 1)
    ypred_registered = np.round(np.exp(registered_model.predict(X_registered)) - 1)
    ypred = ypred_casual + ypred_registered
    return(ypred)


df = pd.read_csv('test_data.csv')
#Basic processing
df = process_data.get_features(df,dummies = False)
#Splitting test data to get the X variables and Y variable
#Note that this step is not needed if when working on daily predictions as that data won't have the Y variables
df.drop(['dteday'],axis = 1, inplace = True)
y_pred = predict(df)
#Save Predictions
pd.DataFrame(y_pred, columns = ['Predicted Cnt']).to_csv('y_pred.csv', index = False)
