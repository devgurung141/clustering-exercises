# imports

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os

import env


def get_zillow():
    '''
    This function reads in zillow data from Codeup database using sql querry into a dataframe and returns a dataframe. 
    '''
        
    query = '''
        SELECT prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        land.propertylandusedesc,
        story.storydesc,
        type.typeconstructiondesc
        FROM properties_2017 prop
        JOIN (
            SELECT parcelid, MAX(transactiondate) AS max_transactiondate
            FROM predictions_2017
            GROUP BY parcelid
            ) pred USING(parcelid)
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
        ''' 
    
    filename = "zillow.csv"
    
    # check if a file exits in a local drive
    if os.path.isfile(filename):
        
        #if yes, read data from a file
        df = pd.read_csv(filename)
        
    else:
        # If not, read data from database into a dataframe
        df = pd.read_sql(query, env.get_connection('zillow'))

        # Write that dataframe in to disk for later, Called "caching" the data for later.
        df.to_csv(filename, index=False)
     
    #returns dataframe
    return df


def row_missing(df):
    num_rows_missing = []
    percent_rows_missing = []
    index = df.columns.tolist()
    for col in df.columns:
        num_rows_missing.append(df[col].isnull().sum())
        percent_rows_missing.append(((df[col].isnull().sum())/len(df))* 100)
        
    metric_df = pd.DataFrame({'num_row_missing': num_rows_missing,'pct_rowss_missing':percent_rows_missing},index=index)
    return metric_df
 
    
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*df.shape[0],0))
#     print(threshold)
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*df.shape[1],0))
#     print(threshold)
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def wrangle_zillow():
    df = get_zillow()
    df = handle_missing_values(df, prop_required_column = .75, prop_required_row = .75)
    return df


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
    