# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:38:58 2020

@author: heman
"""
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pickle
import process_data

def load_data(date):
    #load only data till Aug 31, 2012
    df = pd.read_csv('hour.csv')
    return(df[df['dteday'] < date])

def train_valid_split(date,df):
    #Spit train and valid data 
    train = df[df['dteday'] < date]
    valid = df[df['dteday'] >= date]
    return(train, valid)


def evaluate_model(model, trainX, train_casual, train_registered, validX, valid_casual, valid_registered):
    #Modeling for Casual. Keep peak_casual
    st = datetime.now()
    train_casualX = trainX.drop(['peak_registered'], axis = 1)
    valid_casualX = validX.drop(['peak_registered'], axis = 1)
    
    model.fit(train_casualX, train_casual)
    predy = pd.DataFrame(model.predict(valid_casualX), columns = ['pred_casual'])
    predy['casual'] = round(np.exp(predy['pred_casual']) - 1)
    
    #Modeling for Registered. Keep Registered
    train_registeredX = trainX.drop(['peak_casual'], axis = 1)
    valid_registeredX = validX.drop(['peak_casual'], axis = 1)
    model.fit(train_registeredX, train_registered)
    predy['pred_registered'] = model.predict(valid_registeredX)
    predy['registered'] = round(np.exp(predy['pred_registered']) - 1)
    
    #Final Prediction
    predy['pred_cnt'] = predy['registered'] + predy['casual']
    
    #Evaluate model by using MAD and MAPE
    y_true = np.array(valid_casual + valid_registered)
    y_pred = np.array(predy['pred_cnt'])
    mad = np.mean(np.absolute(y_pred - y_true))
    mape = np.mean(np.abs(np.absolute(y_true - y_pred) / y_true)) * 100
    end = datetime.now()
    time = (end - st).seconds
    return(mad, mape, time)

def compare_models(models):
    #Evaluate each model
    labels = ['RF','GBM', 'LGBM']
    MAD = []

    for model in models:
        MAD.append(evaluate_model(model, trainX, train_casual, train_registered, validX, valid_casual, valid_registered))
    
    
    MAD = pd.DataFrame(MAD, columns = ['Mean Absolute Deviations', 'Mean Absolute Percentage Error', 'Time'])
    MAD['Models'] = labels
    return(MAD)

def init_models():
    #Initialize models
    RF = RandomForestRegressor(random_state=1)
    GBM = GradientBoostingRegressor(random_state=1, learning_rate= 0.02, n_estimators = 1000, max_depth=8)
    LGBM = lightgbm.LGBMRegressor(random_state=1, learning_rate=0.02, n_estimators =1000, max_depth = 8)
    models = [RF,GBM,LGBM]
    return(models)  
        

def save_model(model, df, results):
    #Train the finalized model on the whole dataset and save the model
    train_X, casual_y, registered_y = process_data.xySplit(df, kind = 'train')
    train_casual_X = train_X.drop(['peak_registered'], axis = 1)
    model.fit(train_casual_X, casual_y)
    filename = 'casual_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    train_registered_X = train_X.drop(['peak_casual'], axis = 1)
    model.fit(train_registered_X, registered_y)
    filename = 'registered_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    filename = 'Validation Results.csv'
    results.to_csv(filename)
    
    

load_date = '2012-09-01'
split_date = '2012-05-01'

df = load_data(load_date)

df = process_data.get_features(df,dummies = False)#Feature Engineering

train, valid = train_valid_split(split_date,df)

trainX, train_casual, train_registered = process_data.xySplit(train, kind = 'train')
validX, valid_casual, valid_registered = process_data.xySplit(valid, kind = 'valid')

models = init_models()
results = compare_models(models)

save_model(models[2],df, results)


