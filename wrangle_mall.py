# imports

import pandas as pd
import numpy as np

import os

import env

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_mall_customers():
    '''acquire mall customers data from Codeup database using sql querry into a dataframe and returns a dataframe. '''
    
    # create sql querry
    query = '''
            SELECT  * FROM 
            customers;
            '''

    # acquire the data using sql querry
    df = pd.read_sql(query, env.get_connection('mall_customers'))
    
    # returns a dataframe
    return df 
 
    
def detect_outliers(df):
    '''takes in a dataframe and print outliers of a dataframe'''
    for col in df.select_dtypes(exclude='object'):
        q1, q3 =df[col].quantile([.25, .75])
        iqr = q3 - q1 
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1- 1.5 * iqr
        print(col)
        print(f' upper bound: {upper_bound}, lower bound: {lower_bound}\n') 
        
        
def train_val_test(df, seed=42):
    '''split data into train, validate and test data'''
    
    # split data into 80% train_validate, 20% test
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    
    # split train_validate data into 70% train, 30% validate
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    
    # returns train, validate, test data
    return train, validate, test


def make_dummies(df, target):
    '''takes a dataframe and target and return a dataframe with dummies'''
     # create dummy vairable for categorical variables
    dummies = pd.get_dummies(df[target],drop_first=True)
    
    # concat dummy variables to dataframe
    df = pd.concat([df,dummies],axis=1)
    
    # drop repetative columns
    df = df.drop(columns=[target])
    
    # returns a dataframe
    return df


def wrangle_mall_customers() :
    '''This function acquire data, clean data, split data and returns train, vaidate and test data.'''
    
    # acquire mall customers data
    df = get_mall_customers()
    
    # get encoded mall customers data
    df = make_dummies(df, 'gender')
    
    # split data into train, validate and test data
    train, validate, test= train_val_test(df, seed=42)
    
    # returns train, validate, test data
    return train, validate, test
   
    
def scale_data(train, validate, test, 
               columns_to_scale= ['age', 'annual_income'],
               return_scaler=False):
    ''' 
    Takes in train, validate, and test data and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # use sacaler
    scaler = MinMaxScaler()
    
    # fit scaler
    scaler.fit(train[columns_to_scale])
    
    # apply the scaler
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled